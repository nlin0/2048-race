import numpy as np
import pytest

torch = pytest.importorskip("torch")

from race2048.dqn import DQNAgent, QNetwork, ReplayBuffer
from race2048.env import Game2048Env


def test_q_network_shape():
    q = QNetwork(state_dim=16, action_dim=4, hidden_dim=256)
    x = torch.zeros(32, 16)
    y = q(x)
    assert y.shape == (32, 4)


def test_q_network_deep_layernorm():
    q = QNetwork(hidden_dim=64, num_hidden_layers=3, layer_norm=True)
    y = q(torch.zeros(4, 16))
    assert y.shape == (4, 4)


def test_load_qnetwork_from_checkpoint_cnn(tmp_path):
    from race2048.dqn.qnet import BoardCNNQNetwork, load_qnetwork_from_checkpoint

    net = BoardCNNQNetwork()
    ck = {"policy_state": net.state_dict(), "qnet_config": net.config_dict()}
    path = tmp_path / "cnn.pt"
    torch.save(ck, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    net2 = load_qnetwork_from_checkpoint(loaded, torch.device("cpu"))
    assert net2.config_dict() == net.config_dict()
    y = net2(torch.zeros(2, 16))
    assert y.shape == (2, 4)


def test_load_qnetwork_from_checkpoint_with_config(tmp_path):
    from race2048.dqn.qnet import load_qnetwork_from_checkpoint

    net = QNetwork(hidden_dim=32, num_hidden_layers=2, layer_norm=True)
    ck = {
        "policy_state": net.state_dict(),
        "qnet_config": net.config_dict(),
    }
    path = tmp_path / "c.pt"
    torch.save(ck, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    net2 = load_qnetwork_from_checkpoint(loaded, torch.device("cpu"))
    assert net2.config_dict() == net.config_dict()


def test_nstep_replay_accumulates():
    from race2048.dqn.buffer import NStepReplayBuffer

    buf = NStepReplayBuffer(1000, gamma=0.5, n=3)
    z = np.zeros(16, dtype=np.float32)
    o = np.ones(16, dtype=np.float32)
    m = np.ones(4, dtype=bool)
    buf.push(z, 0, 1.0, o, False, m)
    buf.push(o, 1, 2.0, z, False, m)
    assert len(buf) == 0
    buf.push(z, 2, 4.0, o, True, m)
    assert len(buf) == 1
    dev = torch.device("cpu")
    _s, _a, r, _ns, _d, _nm, boot = buf.sample(1, dev)
    assert r.item() == pytest.approx(3.0)
    assert boot.item() == pytest.approx(0.125)


def test_replay_sample_device():
    buf = ReplayBuffer(1000)
    for i in range(200):
        s = np.random.randn(16).astype(np.float32)
        ns = np.random.randn(16).astype(np.float32)
        mask = np.ones(4, dtype=bool)
        buf.push(s, i % 4, float(i), ns, i % 50 == 0, mask)
    dev = torch.device("cpu")
    states, a, r, ns, d, nm, boot = buf.sample(32, dev)
    assert states.shape == (32, 16)
    assert nm.dtype == torch.bool
    assert nm.shape == (32, 4)
    assert boot.shape == (32,)
    assert torch.allclose(boot, torch.full_like(boot, 0.99))


def test_dqn_loss_step():
    agent = DQNAgent(device="cpu", target_update_every=10_000, eps_decay_steps=1)
    buf = ReplayBuffer(5000)
    for _ in range(300):
        s = np.zeros(16, dtype=np.float32)
        ns = np.ones(16, dtype=np.float32)
        buf.push(s, 0, 0.0, ns, False, np.ones(4, dtype=bool))
    loss = agent.train_step(buf, 64)
    assert loss is not None
    assert loss >= 0.0


def test_replay_q_stats():
    agent = DQNAgent(device="cpu", target_update_every=10_000, eps_decay_steps=1)
    buf = ReplayBuffer(5000)
    for _ in range(300):
        s = np.random.randn(16).astype(np.float32)
        ns = np.random.randn(16).astype(np.float32)
        buf.push(s, 0, 0.0, ns, False, np.ones(4, dtype=bool))
    st = agent.replay_q_stats(buf, 64)
    assert st is not None
    assert "q_sa_mean" in st
    assert "q_abs_max" in st
    assert st["q_abs_max"] >= 0.0
    assert agent.replay_q_stats(buf, 10_000) is None


def test_epsilon_decays():
    a = DQNAgent(device="cpu", eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    assert a.epsilon(0) == pytest.approx(1.0)
    assert a.epsilon(100) == pytest.approx(0.1)
    assert a.epsilon(50) > a.epsilon(100)


def test_epsilon_offset_restarts_schedule():
    a = DQNAgent(device="cpu", eps_start=1.0, eps_end=0.0, eps_decay_steps=100)
    a.set_epsilon_offset(100)
    assert a.epsilon(100) == pytest.approx(1.0)
    assert a.epsilon(200) == pytest.approx(0.0)


def test_select_action_zero_epsilon_only_legal():
    """With ε=0, action must be argmax among legal entries."""
    agent = DQNAgent(device="cpu", eps_decay_steps=100, eps_start=0.0, eps_end=0.0)
    obs = np.zeros((4, 4), dtype=np.float32)
    mask = np.array([False, True, False, False], dtype=np.bool_)
    assert agent.select_action(obs, mask, 0) == 1
    mask2 = np.array([True, True, False, False], dtype=np.bool_)
    a = agent.select_action(obs, mask2, 0)
    assert a in (0, 1)


def test_policy_weights_change_with_training():
    """Grad steps on a non-trivial batch should move policy parameters."""
    agent = DQNAgent(device="cpu", target_update_every=10_000, eps_decay_steps=1)
    buf = ReplayBuffer(5000)
    for i in range(200):
        s = np.random.randn(16).astype(np.float32)
        ns = np.random.randn(16).astype(np.float32)
        buf.push(s, i % 4, float(i % 10), ns, i % 40 == 0, np.ones(4, dtype=bool))
    p0 = next(agent.policy_net.parameters()).detach().clone()
    for _ in range(30):
        agent.train_step(buf, 64)
    p1 = next(agent.policy_net.parameters())
    assert not torch.allclose(p0, p1, rtol=1e-4, atol=1e-5)


def test_train_smoke_short():
    """Few env steps + train step without crash."""
    agent = DQNAgent(device="cpu", target_update_every=5, eps_decay_steps=50)
    buf = ReplayBuffer(5000)
    env = Game2048Env(seed=0, max_steps=500)
    obs, info = env.reset(seed=0)
    for step in range(1, 101):
        flat = np.asarray(obs, dtype=np.float32).reshape(-1)
        act = agent.select_action(obs, info["legal_action_mask"], step)
        next_obs, r, term, trunc, info = env.step(act)
        done = term or trunc
        buf.push(
            flat,
            act,
            float(r),
            np.asarray(next_obs, dtype=np.float32).reshape(-1),
            done,
            info["legal_action_mask"],
        )
        if done:
            obs, info = env.reset(seed=step)
        else:
            obs = next_obs
        if len(buf) >= 64:
            agent.train_step(buf, 64)
