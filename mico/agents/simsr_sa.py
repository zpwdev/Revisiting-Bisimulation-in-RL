import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import itertools


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward2(self, z, std):
        mu = self.policy(z)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        return mu, std

    def forward(self, obs, std):
        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, latent_a_dim, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action, act_tok=None):
        if act_tok is not None:
            action = act_tok(action)
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class SimSR_sa(nn.Module):
    def __init__(self, repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, act_tok, encoder, device):
        super(SimSR_sa, self).__init__()

        self.encoder = encoder
        self.device = device

        a_dim = action_shape[0]

        self.proj_sa = nn.Sequential(
            nn.Linear(feature_dim + latent_a_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.act_tok = act_tok

        self.proj_s = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())

        self.apply(utils.weight_init)

    def encode(self, x, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.proj_s[0](self.encoder(x))
        else:
            z_out = self.proj_s[0](self.encoder(x))
        z_out = nn.functional.normalize(z_out, dim=1, p=2)
        return z_out
    
    """
    def encode_norm(self, x, ema=False):
        if ema:
            with torch.no_grad():
                z_out = self.proj_s[0](self.encoder(x))
                z_out = nn.functional.normalize(z_out, dim=1, p=2)
        else:
            z_out = self.proj_s[0](self.encoder(x))
            z_out = nn.functional.normalize(z_out, dim=1, p=2)
        return z_out
    """

    def project_sa(self, s, a):
        x = torch.concat([s, a], dim=-1)
        x = self.proj_sa(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return x


class SimSR_sa_Agent:
    def __init__(self, obs_shape, action_shape, device, lr, encoder_lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 reward, multistep, latent_a_dim, curl):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        ### A heuristics to choose the dimensionality of latent actions
        if latent_a_dim == 'none':
            latent_a_dim = int(action_shape[0] * 1.25) + 1
        ### Create action embeddings
        self.act_tok = utils.ActionEncoding(action_shape[0], latent_a_dim, multistep)
        self.encoder = Encoder(obs_shape, feature_dim).to(device)

        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)
        self.critic = Critic(self.encoder.repr_dim, latent_a_dim, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, latent_a_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.SimSR_sa = SimSR_sa(self.encoder.repr_dim, feature_dim, action_shape, latent_a_dim, hidden_dim, self.act_tok, self.encoder, device).to(device)

        self.raw_c = torch.tensor([0.0, 0.0]).to(device)
        self.raw_c.requires_grad = True

        ### State & Action Encoders
        parameters = itertools.chain(self.encoder.parameters(),
                                     self.act_tok.parameters(),
                                     )
        self.encoder_opt = torch.optim.Adam(parameters, lr=encoder_lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.SimSR_sa_opt = torch.optim.Adam(self.SimSR_sa.parameters(), lr=encoder_lr)
        self.c_optimizer = torch.optim.Adam([self.raw_c], lr=encoder_lr)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.SimSR_sa.train()

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action, self.act_tok)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action, self.act_tok)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action, self.act_tok)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_simsr_sa(self, obs, action, action_seq, next_obs, reward, discount, step):
        metrics = dict()

        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())

        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)

        z = self.SimSR_sa.encode(obs_en)

        obs_en2 = obs_en[perm]
        with torch.no_grad():
            z2 = self.SimSR_sa.encode(obs_en2, ema=True)
            next_z = self.SimSR_sa.encode(next_obs_en, ema=True)
            next_z2 = next_z[perm]
            reward2 = reward[perm]

        def cosine_dis(feature_a, feature_b, t):
            feature_a_norm = nn.functional.normalize(feature_a, dim=1)
            feature_b_norm = nn.functional.normalize(feature_b, dim=1)
            temp = torch.matmul(feature_a_norm, feature_b_norm.T)
            dis = torch.diag(temp)
            return dis

        def diff(z1, z2, z_cosine_dis_, t, beta=0.1):
            z_cosine_dis_ = torch.clip(z_cosine_dis_, min=0.0001, max=0.9999)
            z_atan2_dis = torch.atan2(torch.sqrt(torch.ones_like(z_cosine_dis_) - torch.pow(z_cosine_dis_, 2)),
                                      z_cosine_dis_)
            z_diffs = 0.5 * (torch.norm(z1, dim=1) + torch.norm(z2, dim=1)) + beta * z_atan2_dis

            return z_diffs

        z_cosine_dis = cosine_dis(z, z2, step)
        z_diff = diff(z, z2, z_cosine_dis, step)
        r_diff = F.smooth_l1_loss(reward, reward2, reduction='none')

        with torch.no_grad():
            z_tar_cosine_dis = cosine_dis(next_z, next_z2, step)
            next_diff = diff(next_z, next_z2, z_tar_cosine_dis, step)

        #c = F.softmax(self.raw_c, dim=0)
        #bisimilarity = c[0] * r_diff + c[1] * next_diff
        bisimilarity = r_diff + discount * next_diff
        encoder_loss = (bisimilarity.detach() - z_diff).pow(2).mean()
        #c_loss = (bisimilarity - z_diff.detach()).pow(2).mean()

        self.SimSR_sa_opt.zero_grad()
        encoder_loss.backward()
        self.SimSR_sa_opt.step()

        #self.c_optimizer.zero_grad()
        #c_loss.backward()
        #self.c_optimizer.step()

        if self.use_tb:
            metrics['z_diff'] = z_diff.mean().item()
            metrics['next_diff'] = next_diff.mean().item()
            metrics['bisimilarity'] = bisimilarity.mean().item()
            metrics['mico_loss'] = encoder_loss.item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        # obs, action, action_seq, reward, discount, next_obs, r_next_obs, r_reward = utils.to_torch(batch, self.device)
        obs, action, action_seq, reward, discount, next_obs, r_next_obs = utils.to_torch(batch, self.device)

        # augment
        obs_en = self.aug(obs.float())
        next_obs_en = self.aug(next_obs.float())
        # encode
        obs_en = self.encoder(obs_en)
        with torch.no_grad():
            next_obs_en = self.encoder(next_obs_en)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs_en, action, reward, discount, next_obs_en, step))

        # update actor
        metrics.update(self.update_actor(obs_en.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        metrics.update(self.update_simsr_sa(obs, action, action_seq, r_next_obs, reward, discount, step))

        return metrics