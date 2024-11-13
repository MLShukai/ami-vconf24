#!/bin/bash
python scripts/launch.py experiment=random_observation_action_log max_uptime=3 && \
python scripts/launch.py experiment=bool_mask_i_jepa_with_videos models=bool_mask_i_jepa_large max_uptime=10 && \
python scripts/launch.py experiment=bool_mask_i_jepa_with_videos models=bool_mask_i_jepa_small max_uptime=10 && \
python scripts/launch.py experiment=learn_only_sioconv models=learn_only_sioconv_large max_uptime=10 && \
python scripts/launch.py experiment=learn_only_sioconv models=learn_only_sioconv_small max_uptime=10 && \
python scripts/launch.py experiment=learn_only_sioconv models=learn_only_sioconv_large max_uptime=10 \
    trainers.forward_dynamics.partial_sampler._target_=ami.trainers.components.random_permutation_sampler.RandomPermutationSampler 
    && \
python scripts/launch.py experiment=learn_i_jepa_sioconv max_uptime=10 && \
python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    && \
python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    && \
python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/NoisyWorld2023/NoisyWorld2023.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    && \
python scripts/launch.py experiment=i_jepa_sioconv_ppo_multi_step \
    max_uptime=10 \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/NoisyWorld2023/NoisyWorld2023.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
