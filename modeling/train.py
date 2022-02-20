"""Here, methods for fine-tuning and the trainer engine are provided.
The code is inspired by Wolf et al. (2019) approach on the PersonaChat Dataset and training TransferTransfo for ConvAI2.
The code is largely adapted from Thomas Wolf's implementation, found here:
https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313
"""

import os
import math
import yaml
from collections import namedtuple
from pprint import pformat
import socket
from datetime import datetime

import torch
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, global_step_from_engine
from ignite.metrics import Accuracy, Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import (WEIGHTS_NAME, CONFIG_NAME, AdamW, GPT2DoubleHeadsModel, AutoTokenizer)

from preprocessing import add_special_token, get_data_loaders


# define special tokens
SPECIAL_TOKENS = ['<bos>', '<eos>', '<speaker1>', '<speaker2>', '<introvert>', '<extrovert>', '<pad>']
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>', '<introvert>', '<extrovert>']}
MODEL_INPUTS = ['input_ids', 'mc_token_ids', 'lm_labels', 'mc_labels', 'token_type_ids']
PADDED_INPUTS = ['input_ids', 'lm_labels', 'token_type_ids']


def get_params(yaml_path):
    """ Load Parameters from yaml config file """
    with open(yaml_path, 'r') as f:
        params_dict = yaml.load(f, Loader=yaml.FullLoader)
    args = namedtuple('Parameters', params_dict.keys())(*params_dict.values())
    return args


def make_logdir(model_name):
    """ Make directory for new training checkpoints """
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + '_' + model_name)
    return log_dir


def average_distributed_scalar(scalar):
    """ Calculate scalar for loss
    Note: distributed setting is not supported on macOS """
    return scalar


def train(config_path, data_path):
    """ Train training on prepared dataset """
    # get parameters from config file
    args = get_params(config_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # initiate pretrained tokenizer and training
    tokenizer = AutoTokenizer.from_pretrained('dbmdz/german-gpt2')
    model_class = GPT2DoubleHeadsModel
    model = model_class.from_pretrained('dbmdz/german-gpt2')
    model.to(device)
    add_special_token(model, tokenizer)
    optimizer = AdamW(model.parameters(), lr=args.lr, correct_bias=True)
    # load data
    train_loader, train_sampler, test_loader, test_sampler = get_data_loaders(data_path, tokenizer, model)

    # initiate trainer and evaluator
    def update(engine, batch):
        """ Calculate loss for language modeling (lm_loss) and next sentence prediction task (mc_loss) """
        model.train()
        batch = tuple(input_tensor.to(device) for input_tensor in batch)
        input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
        (lm_loss), (mc_loss), *_ = model(
            input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            mc_labels=mc_labels)

        loss = (lm_loss * args.lm_coef + mc_loss * args.mc_coef) / args.gradient_accumulation_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)

        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item()
    trainer = Engine(update)


    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = tuple(input_tensor.to(device) for input_tensor in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch
            print(tokenizer.decode(input_ids[0, -1, :].tolist()))

            outputs = model(
                input_ids, token_type_ids=token_type_ids, mc_token_ids=mc_token_ids,
            )
            lm_logits = outputs.logits
            mc_logits = outputs.mc_logits
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return (lm_logits_flat_shifted, mc_logits), (lm_labels_flat_shifted, mc_labels)
    evaluator = Engine(inference)


    # evaluation at starting point of training and at end of each training epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(test_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(test_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(test_loader))

    # linearly decrease lr to zero
    scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    metrics = {'nll': Loss(torch.nn.CrossEntropyLoss(ignore_index=100), output_transform=lambda x: (x[0][0], x[1][0])),
               'accuracy': Accuracy(output_transform=lambda x: (x[0][1], x[1][1]))}

    metrics.update({'average_nll': MetricsLambda(average_distributed_scalar, metrics['nll']),
                    'average_accuracy': MetricsLambda(average_distributed_scalar, metrics['accuracy'])}) #args??
    metrics['average_ppl'] = MetricsLambda(math.exp, metrics['average_nll'])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # add progress bar, checkpoints, etc
    if args.local_rank in [-1, 0]:
        # progress bar
        pbar = ProgressBar(persistent=True)
        pbar.attach(trainer, metric_names=['loss'])
        evaluator.add_event_handler(Events.COMPLETED, lambda _: pbar.log_message(
            'Validation: %s' % pformat(evaluator.state.metrics)))
        # save check points
        log_dir = make_logdir(args.model_checkpoint)
        tb_logger = TensorboardLogger(log_dir)

        tb_logger.attach(trainer, log_handler=OutputHandler(tag='training', metric_names=['loss']),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag='validation', metric_names=list(metrics.keys()),
                                                              global_step_transform=global_step_from_engine(trainer)),
                         event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(log_dir, 'checkpoint', save_interval=1, n_saved=3)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {'mymodel': getattr(model, 'module', model)})
        # save training and config
        #torch.save(args, log_dir + '/model_training_args.bin')
        getattr(model, 'module', model).config.to_json_file(os.path.join(log_dir, CONFIG_NAME))
        tokenizer.save_pretrained(log_dir)

    # run training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # rename the last checkpoint
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(os.path.join(log_dir, checkpoint_handler._saved[-1][1]), os.path.join(log_dir, WEIGHTS_NAME))
        tb_logger.close()


if __name__ == '__main__':
    config_path = 'config.yaml'
    data_path = '../outputs/training/context-chats.csv'
    train(config_path, data_path)

    print("""
    *******************************************************

                    *** Congratulations ***

    You successfully trained your training. Continue with the
    interaction pipeline to chat with it or skip straight 
    ahead to the evaluation. 


    *******************************************************
    """)