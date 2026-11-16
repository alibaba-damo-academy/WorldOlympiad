"""Recovery archive timestamps must be available in the generator runtime."""
from autoresearch import robotics_scenecode_case_generator as generator


def test_recovery_archive_clock_is_bound():
    # The inventory-retry branch calls time.time_ns(); importing the generator
    # alone previously passed while both transfer scene jobs crashed here.
    first = generator.time.time_ns()
    second = generator.time.time_ns()
    assert isinstance(first, int)
    assert second >= first
