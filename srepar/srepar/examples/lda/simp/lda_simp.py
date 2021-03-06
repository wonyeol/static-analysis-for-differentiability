"""
Source: `whitebox/refact/test/pyro_example/lda_simp_1.py`.
Changes from the source: marked as `# WL: ...`.
  + removed all lines depending on the variable `debug`.
"""

# source: http://pyro.ai/examples/lda.html
# replaced some args.* variables into default value.
# args.num_words -> 1024 
# args.num_topics -> 8
# args.num_docs -> 1000
# args.num_words_per_doc -> 64
#
# things to look at 
# - Dirichlet, Delta and from pyro.distributions
# - pyro.sample(..., infer={"enumerate":"parallel"} )    in model              
# - pyro.param("name", lambda: tensor, constraint=constraints.positive)    in guide 
# - (Tensor).scatter_add_(dim, index, other) → Tensor (tensor shape is not changed)
#   ex) counts = counts.scatter_add(0, data[: ,1], t)     in guide                         
 

from __future__ import absolute_import, division, print_function

import argparse
import functools
import logging
# WL: added. =====
import time, sys
# ================

import torch
from torch import nn
from torch.distributions import constraints

import pyro
from pyro.distributions import Gamma, Dirichlet, Categorical, Delta 
import pyro.distributions as dist
from pyro.infer import SVI, JitTraceEnum_ELBO, TraceEnum_ELBO, TraceGraph_ELBO
from pyro.optim import Adam

# WL: edited. =====
# logging.basicConfig(format='%(relativeCreated) 9d %(message)s', level=logging.INFO)
logging.basicConfig(format='%(message)s', level=logging.INFO, stream=sys.stdout)
# =================

#########
# model #
#########
# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model(data=None, args=None, batch_size=None):
    data = torch.reshape(data, [64, 1000])
    
    # Globals.
    with pyro.plate("topics", 8):
        # shape = [8] + []
        topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
        # shape = [8] + [1024]
        topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))

    # Locals.
    # WL: edited. =====
    # with pyro.plate("documents", 1000, 32, dim=-1) as ind:
    with pyro.plate("documents", 1000) as ind:
    # =================
        # if data is not None:
        #     data = data[:, ind]
        # shape = [64, 32]
        data = data[:, ind]
        # shape = [32] + [8]
        doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))

        # WL: edited. =====
        # with pyro.plate("words", 64, dim=-2):
        with pyro.plate("words", 64):
        # =================
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            # shape = [64, 32] + []
            word_topics =\
                pyro.sample("word_topics", Categorical(doc_topics),
                            infer={"enumerate": "parallel"})
                # pyro.sample("word_topics", Categorical(doc_topics))
            # shape = [64, 32] + []
            data =\
                pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                            obs=data)
            
    return topic_weights, topic_words, data

#########
# guide #
#########
# nn
layer1 = nn.Linear(1024,100)
layer2 = nn.Linear(100,100)
layer3 = nn.Linear(100,8)
sigmoid = nn.Sigmoid()
# WL: added. =====
softmax = nn.Softmax(dim=-1)
# ================

layer1.weight.data.normal_(0, 0.001)
layer1.bias.data.normal_(0, 0.001)
layer2.weight.data.normal_(0, 0.001)
layer2.bias.data.normal_(0, 0.001)
layer3.weight.data.normal_(0, 0.001)
layer3.bias.data.normal_(0, 0.001)

# guide
def guide(data, args, batch_size=None):
    data = torch.reshape(data, [64, 1000])
    
    pyro.module("layer1", layer1)
    pyro.module("layer2", layer2)
    pyro.module("layer3", layer3)

    # Use a conjugate guide for global variables.
    topic_weights_posterior = pyro.param(
        "topic_weights_posterior",
        # WL: edited. =====
        # # lambda: torch.ones(8) / 8,
        # torch.ones(8) / 8,
        torch.ones(8),
        # =================
        constraint=constraints.positive)
    topic_words_posterior = pyro.param(
        "topic_words_posterior",
        # WL: edited. =====
        # # lambda: torch.ones(8, 1024) / 1024,
        # torch.ones(8, 1024) / 1024,
        # constraint=constraints.positive)
        torch.ones(8, 1024),
        constraint=constraints.greater_than(0.5))
        # =================
    """
    # wy: dummy param for word_topics
    word_topics_posterior = pyro.param(
        "word_topics_posterior",
        torch.ones(64, 1024, 8) / 8,
        constraint=constraints.positive)
    """
    
    with pyro.plate("topics", 8):
        # shape = [8] + []
        topic_weights = pyro.sample("topic_weights", Gamma(topic_weights_posterior, 1.))
        # shape = [8] + [1024]
        topic_words = pyro.sample("topic_words", Dirichlet(topic_words_posterior))

    # Use an amortized guide for local variables.
    with pyro.plate("documents", 1000, 32) as ind:
        # shape =  [64, 32]
        data = data[:, ind]
        # The neural network will operate on histograms rather than word
        # index vectors, so we'll convert the raw data to a histogram.
        counts = torch.zeros(1024, 32)
        counts = torch.Tensor.scatter_add_\
            (counts, 0, data,
             torch.Tensor.expand(torch.tensor(1.), [1024, 32]))
        h1 = sigmoid(layer1(torch.transpose(counts, 0, 1)))
        h2 = sigmoid(layer2(h1))
        # shape = [32, 8]
        doc_topics_w = sigmoid(layer3(h2))
        # shape = [32] + [8]
        # WL: edited. =====
        # # # doc_topics = pyro.sample("doc_topics", Delta(doc_topics_w, event_dim=1))
        # # doc_topics = pyro.sample("doc_topics", Delta(doc_topics_w).to_event(1))
        # doc_topics = pyro.sample("doc_topics", Dirichlet(doc_topics_w))
        doc_topics = softmax(doc_topics_w)
        pyro.sample("doc_topics", Delta(doc_topics, event_dim=1))
        # =================
        """
        # wy: dummy sample for word_topics
        pyro.sample("word_topics", Categorical(word_topics_posterior[:,ind,:]))
        """

#======================== original
# This is a fully generative model of a batch of documents.
# data is a [num_words_per_doc, num_documents] shaped array of word ids
# (specifically it is not a histogram). We assume in this simple example
# that all documents have the same number of words.
def model_original(data=None, args=None, batch_size=None):
    # Globals.
    with pyro.plate("topics", 8):
        topic_weights = pyro.sample("topic_weights", Gamma(1. / 8, 1.))
        topic_words = pyro.sample("topic_words", Dirichlet(torch.ones(1024) / 1024))
    # Locals.
    with pyro.plate("documents", 1000) as ind:
        if data is not None:
            data = data[:, ind]
        doc_topics = pyro.sample("doc_topics", Dirichlet(topic_weights))
        with pyro.plate("words", 64):
            # The word_topics variable is marginalized out during inference,
            # achieved by specifying infer={"enumerate": "parallel"} and using
            # TraceEnum_ELBO for inference. Thus we can ignore this variable in
            # the guide.
            word_topics =\
                pyro.sample("word_topics", Categorical(doc_topics),
                            infer={"enumerate": "parallel"})
            data =\
                pyro.sample("doc_words", Categorical(topic_words[word_topics]),
                            obs=data)
    return topic_weights, topic_words, data
#======================== original

def main(args):
    logging.info('Generating data')
    # WL: edited. =====
    # pyro.set_rng_seed(0)
    pyro.clear_param_store()
    pyro.enable_validation(__debug__)
    # ====================

    # We can generate synthetic data directly by calling the model.
    true_topic_weights, true_topic_words, data = model_original(args=args)

    # We'll train using SVI.
    logging.info('-' * 40)
    logging.info('Training on {} documents'.format(args.num_docs))
    # wy: currently don't do enumeration.
    # # Elbo = JitTraceEnum_ELBO if args.jit else TraceEnum_ELBO
    Elbo = TraceEnum_ELBO
    # Elbo = TraceGraph_ELBO
    elbo = Elbo(max_plate_nesting=2)
    optim = Adam({'lr': args.learning_rate})
    svi = SVI(model, guide, optim, elbo)

    # WL: edited. =====
    # logging.info('Step\tLoss')
    logging.info(args)
    times = [time.time()]
    logging.info('\nstep\t' + 'epoch\t' + 'elbo\t' + 'time(sec)')
    # =================

    # WL: edited. =====
    # for step in range(args.num_steps):
    for step in range(1, args.num_steps+1):
    # =================
        loss = svi.step(data, args=args, batch_size=args.batch_size)

        # WL: edited. =====
        # if step % 10 == 0:
        #    logging.info('{: >5d}\t{}'.format(step, loss))
        if (step % 10 == 0) or (step == 1):
            times.append(time.time())
            logging.info(f'{step:06d}\t'
                         f'{(step * args.batch_size) / args.num_docs:.3f}\t'
                         f'{-loss:.4f}\t'
                         f'{times[-1]-times[-2]:.3f}')
        # =================

    # WL: commented. =====
    # loss = elbo.loss(model, guide, data, args=args)
    # logging.info('final loss = {}'.format(loss))
    # ====================


if __name__ == '__main__':
    # assert pyro.__version__.startswith('0.3.0')
    parser = argparse.ArgumentParser(description="Amortized Latent Dirichlet Allocation")
    parser.add_argument("-t", "--num-topics", default=8, type=int)
    parser.add_argument("-w", "--num-words", default=1024, type=int)
    parser.add_argument("-d", "--num-docs", default=1000, type=int)
    parser.add_argument("-wd", "--num-words-per-doc", default=64, type=int)
    parser.add_argument("-n", "--num-steps", default=1000, type=int)
    # parser.add_argument("-n", "--num-steps", default=100, type=int)
    parser.add_argument("-l", "--layer-sizes", default="100-100")
    # WL: edited. =====
    # parser.add_argument("-lr", "--learning-rate", default=0.001, type=float)
    parser.add_argument("-lr", "--learning-rate", default=0.01, type=float)
    # =================
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    # parser.add_argument('--jit', action='store_true')
    args = parser.parse_args()
    main(args)
