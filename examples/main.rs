extern crate random;
extern crate tensorflow;

use cosyne::{Activation, Config, Cosyne, Environment, ANN};
use gym_rs::{ActionType, CartPoleEnv, GifRender, GymEnv};
use once_cell::sync::OnceCell;
use std::error::Error;
use std::result::Result;
use std::time::Instant;
use tensorflow::expr::{Compiler, Placeholder};
use tensorflow::Code;
use tensorflow::Graph;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Status;
use tensorflow::Tensor;

fn main() {
    pretty_env_logger::init();

    let config = Config::new(100);
    let env = Box::new(CartPoleEvaluator {});
    let nn = ANN::new(6, 1, Activation::Relu);
    let mut cosyne = Cosyne::new(env, nn, config);
    let t0 = Instant::now();
    for _ in 0..100 {
        cosyne.evolve();
    }
    let champion = cosyne.champion();
    println!("champion: {:?}", champion);
    println!("training time: {}ms", t0.elapsed().as_secs());
    assert!(champion.1 >= 400.0);

    render_champion(&mut champion.0.clone());
}

static GAMMA: u32 = 0.99;
static MAX_STEP: i32 = 200;
static NUM_EPISODE: i32 = 1000;
static BATCH_SIZE: i32 = 32;
static CAPACITY: i32 = 10000;

pub struct ReplayMemory {
    capacity: i32,
    memory: Vec<i32>,
    index: i32,
}

impl ReplayMemory {
    pub fn push(state, action, next_state, reward){
        capacity = *CAPACITY;
    if memory.len() < capacity {
        memory.push(None)}}}

impl Brain {}

impl Agent {}

impl Environment for CartPoleEvaluator {
    fn evaluate(&self, nn: &mut ANN) -> f64 {
        let mut env = CartPoleEnv::default();

        let mut state: Vec<f64> = env.reset();

        let mut end: bool = false;
        let mut total_reward: f64 = 0.0;
        while !end {
            if total_reward >= 400.0 {
                break;
            }
            let output = nn.forward(state);
            let action: ActionType = if output[0] < 0.5 {
                ActionType::Discrete(0)
            } else {
                ActionType::Discrete(1)
            };
            let (s, reward, done, _info) = env.step(action);
            end = done;
            state = s;
            total_reward += reward;
        }
        total_reward
    }
}

fn render_champion(champion: &mut ANN) {
    println!("rendering champion...");

    let mut env = CartPoleEnv::default();

    let mut render = GifRender::new(540, 540, "img/sample.gif", 20).unwrap();

    let mut state: Vec<f64> = env.reset();

    let mut end: bool = false;
    let mut steps: usize = 0;
    while !end {
        if steps > 300 {
            break;
        }
        let output = champion.forward(state);
        let action: ActionType = if output[0] < 0.5 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}
