#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source /home/anirudh/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
conda activate nfl_gan

#env="GravitarNoFrameskip-v4"
#env="SpaceInvadersNoFrameskip-v4"
#env="MsPacmanNoFrameskip-v4"
#env="SeaquestNoFrameskip-v4"
#env="AlienNoFrameskip-v4"
#env="HeroNoFrameskip-v4"
#env="StarGunnerNoFrameskip-v4"
#env="AmidarNoFrameskip-v4"
#env="BreakoutNoFrameskip-v4"
#env="QbertNoFrameskip-v4"
#env="FrostbiteNoFrameskip-v4"
#env="BattleZoneNoFrameskip-v4"
#env="UpNDownNoFrameskip-v4"
#env="VideoPinballNoFrameskip-v4"
#env="BankHeistNoFrameskip-v4"
#env="WizardOfWorNoFrameskip-v4"
#env="ZaxxonNoFrameskip-v4"
#env="AsterixNoFrameskip-v4"
#env="GopherNoFrameskip-v4"
#env="JamesbondNoFrameskip-v4"
#env="KangarooNoFrameskip-v4"


#env="AssaultNoFrameskip-v4"
#env="AsteroidsNoFrameskip-v4"
#env="AtlantisNoFrameskip-v4"
env="BeamRiderNoFrameskip-v4"
#env="BowlingNoFrameskip-v4"
#env="BoxingNoFrameskip-v4"
#env="CentipedeNoFrameskip-v4"
#env="ChopperCommandNoFrameskip-v4"
#env="CrazyClimberNoFrameskip-v4"
#env="DemonAttackNoFrameskip-v4"
#env="DoubleDunkNoFrameskip-v4"
#env="EnduroNoFrameskip-v4"
#env="FishingDerbyNoFrameskip-v4"
#env="FreewayNoFrameskip-v4"
#env="IceHockeyNoFrameskip-v4"
#env="KrullNoFrameskip-v4"
#env="KungFuMasterNoFrameskip-v4"
#env="MontezumaRevengeNoFrameskip-v4"
#env="NameThisGameNoFrameskip-v4"
#env="PongNoFrameskip-v4"
#env="PrivateEyeNoFrameskip-v4"
#env="RiverraidNoFrameskip-v4"
#env="RoadRunnerNoFrameskip-v4"
#env="RobotankNoFrameskip-v4"
#env="TennisNoFrameskip-v4"
#env="TimePilotNoFrameskip-v4"
#env="TutankhamNoFrameskip-v4"
#env="VentureNoFrameskip-v4"
#env="ZaxxonNoFrameskip-v4"


nhid=512
blocks=6
topk=4
log="/home/anirudh/iclr2021/modular_central/blocks_atari/logs/Blocks_"$env"-name_"$nhid"_"$blocks"_"$topk
save="/home/anirudh/iclr2021/modular_central/blocks_atari/trained_models/Blocks_"$env"-name_"$nhid"_"$blocks"_"$topk

python /home/anirudh/iclr2021/modular_central/blocks_atari/main.py --env-name $env --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 8 --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --log-dir $log --save-dir $save
