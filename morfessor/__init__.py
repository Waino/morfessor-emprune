#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Morfessor 2.0 - Python implementation of the Morfessor method
"""
import logging


__all__ = ['MorfessorException', 'ArgumentException', 'MorfessorIO',
           'BaselineModel', 'main', 'get_default_argparser', 'main_evaluation',
           'get_evaluation_argparser']

__version__ = '2.0.7'
__author__ = 'Sami Virpioja, Peter Smit, Stig-Arne Grönroos'
__author_email__ = "morpho@aalto.fi"

show_progress_bar = True

_logger = logging.getLogger(__name__)


def get_version():
    return __version__

# The public api imports need to be at the end of the file,
# so that the package global names are available to the modules
# when they are imported.

from .corpus import FixedCorpusWeight, AnnotationCorpusWeight, \
    NumMorphCorpusWeight, MorphLengthCorpusWeight, \
    AlignedTokenCountCorpusWeight
from .baseline import BaselineModel
from .cmd import main, get_default_argparser, configure_logger, \
    main_evaluation, get_evaluation_argparser
from .exception import MorfessorException, ArgumentException
from .io import MorfessorIO
from .utils import _progress
from .evaluation import MorfessorEvaluation, MorfessorEvaluationResult

from .constructions.base import BaseConstructionMethods
from .constructions.parallel import ParallelConstructionMethods
