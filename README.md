# StoryTeller
The intel OneAPI based storyteller is an interactive way to listen the things that you want and have waht you want at your fingertips fast and precise, this is a AI based chatbot system which can generate images by using the Intel's API toolkit which provide parallelism and efficieny to it. So, let's see what we got in here.

##Installing the necessary libraries

<pre>
    ```python
!pip install gTTS moviepy diffusers

!pip install torch torchvision

!pip install diffusers["torch"] transformers

!pip install accelerate

!pip install git+https://github.com/huggingface/diffusers
    ```
    </pre>


##Story Creation
<pre>
```python
from transformers import pipeline, set_seed

import intel_extension_for_pytorch as ipex

Set up the pipeline

pipe = pipeline("text-generation", model="openai-community/gpt2")

Set the seed for reproducibility

set_seed(42)

Define the text prompt
text = " a boy in the forest"
prompt = "generate a story on the title" + text
{Set the maximum number of tokens}
max_length = 1024


###Generate the story

story = pipe(
    prompt,
    max_length=max_length,
    truncation=True
)[0]['generated_text'][len(prompt)+2:]
print(story)```
    </pre>
from gtts import gTTS
narration_text = story
narration = gTTS(text=narration_text, lang='en-us', slow=True, tld='com')
narration.save("narration.mp3")
audio = AudioFileClip("narration.mp3")
duration = audio.duration
import torch
from diffusers import StableDiffusionPipeline

# Load the model
pipe = StableDiffusionPipeline.from_pretrained("digiplay/majicMIX_realistic_v6", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.safety_checker = None

h = 800  # height of the image
w = 640  # width of the image
steps = 25  # number of updates the system makes before giving the result, making it more accurate
guidance = 7.5  # how closely you want the image to be related to the prompt that you have typed
neg = "easynegative,no repetation, lowres,partial view, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot,"

prompt=text
num_images=int(duration/3)
for i in range(num_images):
    prompt = prompt
    image = pipe(prompt, height=h, width=w, number_inference_steps=steps, guidance_scale=guidance, negative_prompt=neg).images[0]
    image.save(f"image_{i+1}.png")  # Save the image with a unique name
