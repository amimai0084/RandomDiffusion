from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline
import torch
import datetime
import random

class RandomDiffusion:
  """
  diffusersで呪文をランダムに試行するモジュール

  環境準備）
    以下コマンドでdiffusersをインストールして下さい。
    pip install --upgrade git+https://github.com/huggingface/diffusers.git transformers accelerate scipy
  """

  def __init__(self, model, output_dir):
    """
    コンストラクタ
    """
    self._model = model
    self._output_dir = output_dir
    self._repeat_cnt = 3    #同じ呪文を繰り返す回数
    self._try_cnt = 10      #異なる呪文を試行する回数
    self._prompt_candidate_list = []  #プロンプトの単語リスト
    self._negative_prompt = 'deformed, blurry, bad anatomy, bad pupil, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, bad hands, fused fingers, messy drawing, broken legs censor, low quality, mutated hands and fingers, long body, mutation, poorly drawn, bad eyes, ui, error, missing fingers, fused fingers, one hand with more than 5 fingers, one hand with less than 5 fingers, one hand with more than 5 digit, one hand with less than 5 digit, extra digit, fewer digits, fused digit, missing digit, bad digit, liquid digit, long body, uncoordinated body, unnatural body, lowres, jpeg artifacts, 3d, cg, text, japanese kanji'
    
  @property
  def repeat_cnt(self):
    return self._repeat_cnt

  @repeat_cnt.setter
  def repeat_cnt(self, value):
    self._repeat_cnt = value
        
  @property
  def try_cnt(self):
    return self._try_cnt

  @try_cnt.setter
  def try_cnt(self, value):
    self._try_cnt = value

  @property
  def negative_prompt(self):
    return self._negative_prompt

  @negative_prompt.setter
  def negative_prompt(self, value):
    self._negative_prompt = value
    
  def addPromptCandidate(self, candidate, max_choice=1, appearance_rate=1):
    """
    プロンプトの候補を追加する
    """
    self._prompt_candidate_list.append({'candidate':candidate, 
              'max_choice':max_choice, 'appearance_rate':appearance_rate})

  def exec(self, model = None):
    if model is not None:
      self._model = model
    try:
      for i in range(0, self._try_cnt):
        self.exec_sub()
    except Exception as e:
      print(f'{self._model} エラー 詳細：{e}')
      
  def exec_sub(self):
    prompt = ''
    for prompt_candidate in self._prompt_candidate_list:
      if random.random() <= prompt_candidate['appearance_rate']: 
        words = random.sample(prompt_candidate['candidate'], 
                  random.randint(1, prompt_candidate['max_choice']))
        if prompt:
          prompt = prompt + ','
        prompt = prompt + ','.join(words)
    print(prompt)
    
    for i in range(0, self._repeat_cnt):
      if 'stablediffusionapi/' in self._model:
        pipe = StableDiffusionPipeline.from_pretrained(self._model, 
                  torch_dtype=torch.float16, use_safetensors=True,)
        pipe.enable_attention_slicing()
      else:
        pipe = DiffusionPipeline.from_pretrained(self._model)
      pipe = pipe.to("cuda")
      image = pipe(prompt, negative_prompt=self._negative_prompt, num_inference_steps=20).images[0]
      file_name = self.make_file_name()
      print(f'{file_name} <= "{prompt}"')
      with open(f"{self._output_dir}/promt.log", "a") as f:  # "a" モードで追記
        f.write(f'{file_name} <= "{prompt}"\n')  # 改行コードを追加
      image.save(file_name)


  def make_file_name(self):
    name = self._model.split('/')[-1]
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    now = datetime.datetime.now(JST)
    return f"{self._output_dir}/{now.strftime('%Y%m%d%H%M%S')}_{name}.png"

