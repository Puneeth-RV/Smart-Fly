"""Smart-Fly: a fruit fly's visual system, from the real connectome, playing Super Mario Bros.

The eye and optic lobe (92,743 neurons) are simulated from the MaleCNS v1.0
wiring diagram and stay fixed. A small network, trained with PPO, turns their
activity into button presses.

Modules, in the order a frame travels through them:

    game        the NES emulator, and the reduction of a frame to the retina
    eye         where each photoreceptor looks on the screen
    brain       graded (non-spiking) neurons on the connectome's wiring
    fly         eye + brain + the features a policy reads
    policy      the trained network, its memory, and loading a trained fly
    train       PPO training
    campaign    working through all 32 levels, level by level
    evaluate    top-choice play with sight intact, removed or scrambled
    probe       what each layer of the visual system still encodes
    web         the live site
"""

__version__ = "1.0.0"
