#!/bin/bash

# Collecting observation and action data
python scripts/launch.py \
    experiment=random_observation_action_log \
    max_uptime=86400 # 24 hours


# Learn I-JEPA
# Please edit `/interaction/environment/folder_paths` in
# configs/experiment/bool_mask_i_jepa_with_videos.yaml

# Large size
python scripts/launch.py \
    experiment=bool_mask_i_jepa_with_videos \
    models=bool_mask_i_jepa_large \
    max_uptime=86400 # 24 hours

# Small size
python scripts/launch.py \
    experiment=bool_mask_i_jepa_with_videos \
    models=bool_mask_i_jepa_small \
    max_uptime=86400 # 24 hours


# Learn SioConv
# Please edit `/interaction/environment/folder_paths` in
# configs/experiment/learn_only_sioconv.yaml

# Large size
python scripts/launch.py \
    experiment=learn_only_sioconv \
    models=learn_only_sioconv_large \
    max_uptime=86400 # 24 hours

# Small size
python scripts/launch.py \
    experiment=learn_only_sioconv \
    models=learn_only_sioconv_small \
    max_uptime=86400 # 24 hours

# Random permutation
python scripts/launch.py \
    experiment=learn_only_sioconv \
    models=learn_only_sioconv_large \
    trainers.forward_dynamics.partial_sampler._target_=ami.trainers.components.random_permutation_sampler.RandomPermutationSampler \
    max_uptime=86400 # 24 hours

# With I-JEPA
python scripts/launch.py \
    experiment=learn_i_jepa_sioconv \
    max_uptime=86400 # 24 hours


# Learn in UnityEnvironment
# Build AMIUnityEnvironment and place the suitable executable path.
SIMPLE_WORLD_PATH="/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64"
NOISY_WORLD_PATH="/workspace/unity_executables/NoisyWorld2023/NoisyWorld2023.x86_64"

# Simple World, Large size.
python scripts/launch.py \
    experiment=i_jepa_sioconv_ppo_multi_step \
    interaction/environment=unity \
    interaction.environment.file_path=/workspace/unity_executables/SimpleWorld/SimpleWorld.x86_64 \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    max_uptime=18000 # 5 hours

# Simple World, Small size.
python scripts/launch.py \
    experiment=i_jepa_sioconv_ppo_multi_step \
    interaction/environment=unity \
    interaction.environment.file_path=$SIMPLE_WORLD_PATH \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    max_uptime=18000 # 5 hours

# Noisy World, Large size.
python scripts/launch.py \
    experiment=i_jepa_sioconv_ppo_multi_step \
    interaction/environment=unity \
    interaction.environment.file_path=$NOISY_WORLD_PATH \
    models=i_jepa_sioconv_resnetpolicy_large \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    max_uptime=18000

# Noisy World, Small size.
python scripts/launch.py \
    experiment=i_jepa_sioconv_ppo_multi_step \
    interaction/environment=unity \
    interaction.environment.file_path=$NOISY_WORLD_PATH \
    models=i_jepa_sioconv_resnetpolicy_small \
    models.policy_value.model.policy_head.action_choices_per_category.path=ami.interactions.environments.actuators.vrchat_osc_discrete_actuator.ACTION_CHOICES_PER_CATEGORY_MASKED_JUMP_RUN \
    max_uptime=18000 # 5 hours
