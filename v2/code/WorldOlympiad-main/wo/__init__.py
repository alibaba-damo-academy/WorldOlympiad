"""WorldOlympiad core: typed task specs, simulation, predicates and the expert.

Layout:
    wo.spec        TaskSpec (objects, steps, goal predicates) and legacy loaders
    wo.scene       resolve a SceneCode generation into simulator inputs
    wo.sim         PyBullet world, robot model and renderer-compatible recording
    wo.predicates  one predicate library with one threshold table
    wo.expert      grasp/goal sampling, motion planning and monitored execution
"""
