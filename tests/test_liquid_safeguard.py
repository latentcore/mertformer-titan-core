from utils.liquid_safeguard import update_liquid_spike_state


def test_liquid_spike_counter_resets_on_safe_loss():
    counter, frozen, triggered = update_liquid_spike_state(
        loss_value=0.9,
        threshold=5.0,
        counter=2,
        patience=3,
        frozen_until=0,
        global_step=10,
        cooldown_steps=50,
        enabled=True,
    )
    assert counter == 0
    assert frozen == 0
    assert triggered is False


def test_liquid_spike_triggers_freeze_after_patience():
    counter = 0
    frozen = 0
    triggered = False
    for step in range(1, 4):
        counter, frozen, triggered = update_liquid_spike_state(
            loss_value=6.0,
            threshold=5.0,
            counter=counter,
            patience=3,
            frozen_until=frozen,
            global_step=step,
            cooldown_steps=25,
            enabled=True,
        )
    assert triggered is True
    assert counter == 0
    assert frozen == 3 + 25


def test_liquid_spike_disabled_noop():
    counter, frozen, triggered = update_liquid_spike_state(
        loss_value=9.0,
        threshold=5.0,
        counter=1,
        patience=3,
        frozen_until=0,
        global_step=5,
        cooldown_steps=10,
        enabled=False,
    )
    assert (counter, frozen, triggered) == (1, 0, False)
