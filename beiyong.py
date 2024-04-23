def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


"""
环境
transformers==4.4.1
transformers包中的file_utils应当添加add_start_docstrings_to_callable方法
具体见本文件开头的方法
此方法是在原transformers==3.4.0中存在的，因此我直接拷到了4.4.1下的file_utils
（虽然不知道此方法具体用处）

file_utils.py226行有问题，
因此最好将环境中的file_utils.py换成本地的file_utils.py文件

**********************************************configuration_bart.py**********************************************
173行
self.static_position_embeddings=True,  # 添加上去的
self.do_blenderbot_90_layernorm = True # 添加上去的
必须添加

**********************************************bart_wang.py*******************************************************
所有的Tokenizer都需要改成BertTokenizer

line 40, 57


**********************************************modeling_bart.py***************************************************
所有config.max_position_embeddings全部改成514，不改的话应该默认是512
不改会报以下错误：
RuntimeError: Error(s) in loading state_dict for BartModel:
size mismatch for encoder.embed_positions.weight: copying a param with shape torch.Size([514, 768]) 
from checkpoint, the shape in current model is torch.Size([512, 768]).
size mismatch for decoder.embed_positions.weight: copying a param with shape torch.Size([514, 768]) 
from checkpoint, the shape in current model is torch.Size([512, 768]).

line 1533/1534 
    out[:, 0: dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
    out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    加上:
    with torch.no_grad():
**********************************************pipe.py************************************************************
1.line 23 tokenizer='bert-base-chinese'
2.line 27 使用BertTokenizer
3.line 80中 word_bpes = [[self.tokenizer.bos_token_id]]，
        改成 word_bpes = [[0]]，因为Bert的tokenizer的bos_token_id==None
4.line word_bpes.append([self.tokenizer.eos_token_id])
        改成 word_bpes.append([1])
    3，4不改的话，数据加载会报错
    3、4和改成什么待定，应为BertTokenizer中0,1不是<s>和</s>    
**********************************************tokenization_bert.py********************************************** 
170,171行，添加的
        bos_token="[CLS]",
        eos_token="[SEP]",
185, 186行，添加的
        bos_token=bos_token,
        eos_token=eos_token,
"""
