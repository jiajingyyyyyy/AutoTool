import numpy as np
from typing import List, Union, Dict, Any
import os
import pickle
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
import time
import sys
from ..config import Config
# 更新可用模型列表，指向SimCSE模型
available_models = ["princeton-nlp/sup-simcse-roberta-base", "princeton-nlp/sup-simcse-bert-base-uncased"]
import atexit


class ToolEmbeddingService:
    """工具嵌入服务，负责生成和管理工具的语义向量表示"""
    def __init__(self, model_name=None, cache_dir=None):
        """
        初始化嵌入服务
        Args:
            model_name: SimCSE模型路径（默认从配置读取）
            cache_dir: 缓存目录（默认从配置读取）
        """
        self.model_name = model_name or Config.SIMCSE_MODEL_PATH
        self.cache_dir = cache_dir or Config.TRANSFORMERS_CACHE_DIR
        
        print(f"加载模型: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            print(f"分词器已加载: {self.model_name}")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            print(f"模型已加载: {self.model_name}")
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
        self.model.eval()
        
        # GPU 支持
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"模型已加载到设备: {self.device}")
        self.vector_cache = {}
        
        safe_model_name_for_cache = os.path.basename(self.model_name).replace('-', '_').replace('/', '_')
        self.cache_path = os.path.join(os.path.dirname(__file__), f"embedding_cache_transformers_{safe_model_name_for_cache}.pkl")
        self.vector_dim = 768  # Default, will be updated after model load
        self._load_cache()
        atexit.register(self._save_cache_on_exit) # 注册退出时保存的函数
    
    def _load_model(self):
        """延迟加载SimCSE模型和分词器 (使用 Transformers)"""
        if self.model is None or self.tokenizer is None:
            print(f"尝试从以下位置加载SimCSE模型和分词器: {self.model_name}")
            try:
                print(f"  使用 transformers.AutoTokenizer.from_pretrained 加载分词器...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
                print(f"  使用 transformers.AutoModel.from_pretrained 加载模型...")
                self.model = AutoModel.from_pretrained(self.model_name, cache_dir=self.cache_dir)
                
                self.model.to(self.device)
                self.model.eval() 

                test_input_for_dim_check = self.tokenizer("Dimension Check", padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    test_output = self.model(**test_input_for_dim_check, output_hidden_states=True, return_dict=True).pooler_output
                self.vector_dim = test_output.shape[-1]
                print(f"SimCSE模型 (via Transformers) 加载成功，向量维度: {self.vector_dim}")
            except OSError as e:
                print(f"SimCSE模型加载失败 (OSError): {e}")
                print(f"  请检查路径 '{self.model_name}' 是否是一个有效的本地模型目录 (应包含config.json等文件)，或者网络连接是否正常。")
                self.model = None
                self.tokenizer = None
            except Exception as e:
                print(f"SimCSE模型加载时发生未知错误: {e}")
                self.model = None
                self.tokenizer = None
    
    def _load_cache(self):
        """加载缓存的嵌入向量"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.vector_cache = pickle.load(f)
                print(f"Loaded {len(self.vector_cache)} cached embeddings")
            except Exception as e:
                print(f"Error loading embedding cache: {e}")
                self.vector_cache = {}

    def _save_cache_on_exit(self):
        print("Attempting to save embedding cache on exit...")
        self._save_cache() # 调用你现有的 _save_cache 方法
    
    def _save_cache(self):
        """保存嵌入向量缓存"""
        try:
            # 确保目录存在
            Path(os.path.dirname(self.cache_path)).mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.vector_cache, f)
            print(f"Saved {len(self.vector_cache)} embeddings to cache")
        except Exception as e:
            print(f"Error saving embedding cache: {e}")
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的嵌入向量，优先使用缓存"""
        if not text:
            return [0.0] * self.vector_dim
        
        cache_key = hash(text)
        
        if cache_key in self.vector_cache:
            return self.vector_cache[cache_key]
        
        if self.model is None or self.tokenizer is None:
            start_time = time.time()
            self._load_model()
            end_time = time.time()
            print(f"模型加载时间: {end_time - start_time:.4f} 秒")
            if self.model is None or self.tokenizer is None: # 再次检查模型加载是否成功
                print("错误: 模型或分词器未能加载，无法生成嵌入。")
                return [0.0] * self.vector_dim
        
        try:
            cleaned_text = text.strip()
            inputs = self.tokenizer(cleaned_text, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                vector_tensor = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            
            vector = vector_tensor.squeeze().cpu().numpy()
            
            vector_list = vector.tolist() if isinstance(vector, np.ndarray) else vector
            
            norm = np.linalg.norm(vector_list)
            if norm > 0:
                vector_list = [v / norm for v in vector_list]
            
            if cache_key not in self.vector_cache: # 只有新计算的才可能需要保存
                self.vector_cache[cache_key] = vector_list
                # 不需要在这里立即保存
            return self.vector_cache[cache_key]
    
        except Exception as e:
            print(f"Error generating SimCSE (via Transformers) embedding for text: {text[:50]}... Error: {e}")
            return [0.0] * self.vector_dim
    
    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取嵌入向量"""
        cache_keys = [hash(text) for text in texts]
        
        results = [None] * len(texts)
        missing_indices = []
        missing_texts_for_model = []

        for i, key in enumerate(cache_keys):
            if key in self.vector_cache:
                results[i] = self.vector_cache[key]
            else:
                missing_indices.append(i)
                missing_texts_for_model.append(texts[i])
        
        if not missing_indices:
            return results

        if self.model is None or self.tokenizer is None:
            self._load_model()
            if self.model is None or self.tokenizer is None:
                print("错误: 模型或分词器未能加载，无法生成批量嵌入。")
                for i in missing_indices: 
                    results[i] = [0.0] * self.vector_dim
                return results
        
        try:
            if missing_texts_for_model:
                inputs = self.tokenizer(missing_texts_for_model, padding=True, truncation=True, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    batch_vectors_tensor = self.model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
                
                batch_vectors = batch_vectors_tensor.cpu().numpy()

                for original_idx_in_missing_list, vec_np in enumerate(batch_vectors):
                    original_text_idx = missing_indices[original_idx_in_missing_list]
                    vector_list = vec_np.tolist()
                    norm = np.linalg.norm(vector_list)
                    if norm > 0:
                        vector_list = [v / norm for v in vector_list]
                    
                    self.vector_cache[cache_keys[original_text_idx]] = vector_list
                    results[original_text_idx] = vector_list
            
            if missing_indices: 
                self._save_cache()
            
            return results
        except Exception as e:
            print(f"Error in SimCSE (via Transformers) batch embedding: {e}")
            for i in range(len(results)):
                if results[i] is None:
                    results[i] = [0.0] * self.vector_dim
            return results
    
    def compute_similarity(self, vec1: Union[List[float], np.ndarray], 
                           vec2: Union[List[float], np.ndarray]) -> float:
        """计算两个向量的余弦相似度"""
        if not isinstance(vec1, np.ndarray):
            vec1 = np.array(vec1)
        if not isinstance(vec2, np.ndarray):
            vec2 = np.array(vec2)
        
        # 处理零向量
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        # 计算余弦相似度
        try:
            # 如果向量已经归一化，直接计算点积即可
            return float(np.dot(vec1, vec2))
        except Exception as e:
            print(f"计算相似度出错: {e}, vec1形状: {vec1.shape}, vec2形状: {vec2.shape}")
            # 使用更稳健的方式计算
            try:
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    return dot_product / (norm1 * norm2)
                return 0.0
            except:
                return 0.0

_embedding_service = None

def get_embedding_service():
    """获取全局嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = ToolEmbeddingService()
    return _embedding_service

def get_embedding(text: str) -> List[float]:
    """获取文本的嵌入向量"""
    service = get_embedding_service()
    return service.get_embedding(text)

def compute_similarity(vec1: Union[List[float], np.ndarray], 
                       vec2: Union[List[float], np.ndarray]) -> float:
    """计算两个向量的余弦相似度"""
    service = get_embedding_service()
    return service.compute_similarity(vec1, vec2)
