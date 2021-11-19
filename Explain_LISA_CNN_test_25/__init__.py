import sys
sys.path
sys.executable
import pip._internal as pip
pip.main(['install','shap'])
pip.main(['install','lime'])
pip.main(['install','alibi'])
from Explain_LISA_CNN_test_25.ExplainLISA import ExplainLISA,Explanations



    