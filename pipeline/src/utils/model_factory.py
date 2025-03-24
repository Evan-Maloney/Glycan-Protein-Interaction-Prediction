from typing import Dict, Any

from ..models.glycan_encoders.dummy import DummyGlycanEncoder
from ..models.glycan_encoders.chemberta import ChemBERTaEncoder
from ..models.glycan_encoders.rdkit import RDKITGlycanEncoder
from ..models.glycan_encoders.gnn import GNNGlycanEncoder

from ..models.protein_encoders.dummy import DummyProteinEncoder
from ..models.binding_predictors.dummy import DummyBindingPredictor
from ..models.protein_encoders.biopy import BioPyProteinEncoder

from ..models.binding_predictors.dnn import DNNBindingPredictor
from ..models.protein_encoders.esmc import ESMCEncoder

from ..models.protein_encoders.pt_gnn import AdvancedGNNProteinEncoder

from ..models.binding_predictors.mean_predictor import MeanPredictor
from ..models.binding_predictors.xgb import XGBBindingPredictor


def create_glycan_encoder(encoder_type: str, **kwargs) -> Any:
    """Create glycan encoder instance based on type"""
    encoders = {
        'dummy': DummyGlycanEncoder,
        'chemberta': ChemBERTaEncoder,
        'rdkit': RDKITGlycanEncoder,
        'gnn': GNNGlycanEncoder
    }
    
    encoder = encoders[encoder_type]
    return encoder(**kwargs)


def create_protein_encoder(encoder_type: str, **kwargs) -> Any:
    encoders = {
        'dummy': DummyProteinEncoder,
        'esmc': ESMCEncoder,
        'biopy': BioPyProteinEncoder,
        'pt_gnn': AdvancedGNNProteinEncoder
    }

    encoder = encoders[encoder_type]
    return encoder(**kwargs)
    
    
def create_binding_predictor(predictor_type: str, **kwargs) -> Any:
    predictors = {
        'dummy': DummyBindingPredictor,
        'dnn': DNNBindingPredictor,
        'mean': MeanPredictor,
        'xgb': XGBBindingPredictor
    }
    
    predictor = predictors[predictor_type]
    return predictor(**kwargs)