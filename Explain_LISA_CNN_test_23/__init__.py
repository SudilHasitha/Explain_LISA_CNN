import sys
sys.path
sys.executable
from Explain_LISA_CNN_test_23.ExplainLISA import ExplainLISA
from Explain_LISA_CNN_test_23.Explanations import Explanations
import pip._internal as pip

pip.main(['install','shap'])
pip.main(['install','lime'])
pip.main(['install','alibi'])