from src.environment import MedicalTriageEnv
from src.models import TriageAction, TriageObservation, TriageReward


def test_reset_returns_valid_observation():
	env = MedicalTriageEnv(max_steps=10, random_seed=42)
	obs = env.reset()

	assert isinstance(obs, TriageObservation)
	assert env.step_count == 0
	assert len(obs.waiting_patients) >= 1
	assert obs.max_steps == 10


def test_step_returns_openenv_tuple_and_updates_state():
	env = MedicalTriageEnv(max_steps=10, random_seed=42)
	obs = env.reset()
	assert obs.waiting_patients

	patient = obs.waiting_patients[0]
	action = TriageAction(patient_id=patient.id, esi_level=3)

	next_obs, reward, done, info = env.step(action)

	assert isinstance(next_obs, TriageObservation)
	assert isinstance(reward, TriageReward)
	assert isinstance(done, bool)
	assert isinstance(info, dict)
	assert info["step"] == 1
	assert env.step_count == 1
	assert 0.0 <= reward.total <= 1.0


def test_state_contains_required_keys():
	env = MedicalTriageEnv(max_steps=5, random_seed=99)
	env.reset()
	state = env.state()

	for key in [
		"step_count",
		"current_time",
		"patients",
		"patient_queue",
		"metrics",
		"episode_start_time",
		"last_action_reward",
	]:
		assert key in state

