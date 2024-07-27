docker build --build-arg CUDA=11.8.0 --build-arg TARGET=cudnn8-devel --build-arg DIST=ubuntu20.04 . -t long-form-factuality
beaker image delete alrope/long-form-factuality
beaker image create long-form-factuality -n long-form-factuality -w ai2/tulu3-factuality 

beaker session create --budget ai2/oe-adapt --image beaker://alrope/long-form-factuality --workspace ai2/pradeepd-open-instruct --secret-env OPENAI_API_KEY=openai_api_key --bare

cd /net/nfs.cirrascale/allennlp/xinxil/factuality-check/long_form_factuality

python validate_hullicination.py --result_path ../results_is_factual/results_ist_rsp.json --n 50 --use_existing_cache