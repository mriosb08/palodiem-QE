#University of Leeds
##Palodiem Project

Transfer-based Quality Estimation (QE).

Scripts to test different trasnfer learning methods on QE.

*Please update paths accordingly to your system.

##Directories

Scripts: (src/) Transfer-based QE scripts.

Data: (data/) QuEst features and HTER labels frm Autodesk data.


##Features

QuEst:
*Dependency: http://www.quest.dcs.shef.ac.uk/

*config for QueEst example: config_en-es.autodesk.properties

Extract QuEst features:

sh extract_quest.sh 

Bicvm, bilingual embeddings:
*dependency: https://github.com/karlmoritz/bicvm
*train bilingual embeddings example:

/data/mrios/workspace/sw/bicvm/bin/dbltrain --input1 autodesk.pt-es.clean.pt --input2 autodesk.pt-es.clean.es --tree plain --type additive --method adagrad --word-width 100 --hinge_loss_margin 128 --model1-out modelA.pt-es.autodesk.100 --model2-out modelB.pt-es.autodesk.100 --noise 10 --batches 20 --eta 0.05 --lambdaD 1 --calc_bi_error1 true --calc_bi_error2 true --iterations 10 &>dblog.pt-es.log

extract bilingual embeddings features:

sh bicvm.sh

##Available transfer models

Self-Taugh Learning (STL):

python src/stlSVR.py 
  --training-examples autodesk.training.en-es.feat 
  --training-labels autodesk.training.en-es.hter 
  --unlabelled-examples autodesk.training.en-pt.feat 
  --test autodesk.test.en-pt.feat 
  --output autodesk.en-pt.pred 
  --epsilon 41.06 
  --c 0.232 
  --hidden-layer 50

Spectral Learning:

usage:python ccaQEpy [training-features] [training-label] [test-features] [test-labels] [U-file] [hid-size]

python /data/mrios/workspace/palodiem-qe/ccaQE.py sent.training.autodesk.en-ru.vec autodesk.training.en-ru.hter sent.test.autodesk.en-pl.vec autodesk.test.en-pl.hter sent.training.autodesk.en-pl.vec 10



SRV Baseline:

usage:python SVR.py [training-features] [training-label] [test-features] [test-labels] [C] [epsilon]

python /data/mrios/workspace/palodiem-qe/SVR.py sent.training.autodesk.en-es.pe.quest autodesk.training.en-es.hter sent.test.autodesk.en-pt-es.quest autodesk.test.en-pt-es.clean.goo.hter 41.06 0.232

Latent variable Gaussian Process (LVGP):

*We use the LVGP instead of autoencoder to perfom STL.

usage:python GPReg.3.py [training-features] [training-label] [test-features] [test-label] [u-file] [hid-dim] [output]

python /data/mrios/workspace/palodiem-qe/GPReg.3.py ../autodesk/sent.training.autodesk.en-es.pe.quest ../autodesk/autodesk.training.en-es.hter wmt12.test.en-es.quest wmt12.test.en-es.goo.hter wmt12.training.en-es.quest 10 deepGP.pred


Transfer SVM (classification only):

sh src/tsvm.sh
