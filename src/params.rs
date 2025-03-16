use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| {
            for i in safetensor.tensors() {
                if i.0 == name {
                    let mut t: Tensor<f32> = Tensor::default(&Vec::from(i.1.shape()));
                    unsafe {
                        for j in 0..t.data().len() {
                            t.data_mut()[j] = f32::from_ne_bytes([i.1.data()[j * 4], i.1.data()[j * 4 + 1], i.1.data()[j * 4 + 2], i.1.data()[j * 4 + 3]]);
                        }
                    }
                    return Some(t);
                }
            }
            return None;
        };

        let get_tensor_model_layers = |name: &str| {
            let mut v = Vec::new();
            for i in 0..config.num_hidden_layers {
                v.push(get_tensor(&format!("model.layers.{}.{}", i, name))?);
            }
            Some(v)
        };

        LLamaParams {
            embedding_table: get_tensor("lm_head.weight").unwrap(), // TODO: need to limit its range
            rms_att_w: get_tensor_model_layers("input_layernorm.weight").unwrap(),
            wq: get_tensor_model_layers("self_attn.q_proj.weight").unwrap(),
            wk: get_tensor_model_layers("self_attn.k_proj.weight").unwrap(),
            wv: get_tensor_model_layers("self_attn.v_proj.weight").unwrap(),
            wo: get_tensor_model_layers("self_attn.o_proj.weight").unwrap(),
            rms_ffn_w: get_tensor_model_layers("post_attention_layernorm.weight").unwrap(),
            w_up: get_tensor_model_layers("mlp.up_proj.weight").unwrap(),
            w_gate: get_tensor_model_layers("mlp.gate_proj.weight").unwrap(),
            w_down: get_tensor_model_layers("mlp.down_proj.weight").unwrap(),
            rms_out_w: get_tensor("model.norm.weight").unwrap(),
            lm_head: get_tensor("lm_head.weight").unwrap(),
        }
    }
}
