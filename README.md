Code implementaion of ICME 2025 InterID: Improving Multi-ID Interaction for Personalized Image Generation. 


**#train**

（1）personalization module:

base model: SD v1.5

bash train.sh

（2）LLM-based Prior Extractor:

base model: Llama 7B

cd LLM-based_Prior_Extractor

bash train_llama_7b_continue_novel.sh

**#inference**

python inference.py

**our checkpoint**

链接: https://pan.baidu.com/s/1jqH8QFPigaDGWZvkii36Pg 提取码: 4wgi
