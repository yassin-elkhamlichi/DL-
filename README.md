# DL
-> To get those numbers into the matrices , we use a two step process : Tokinization and embedding
- Before we even touch NN , we need a Vocabulary(Dictionary)
-- Each unique word in your data is assigned a specific index :
    yassin -> 104
    is -> 25
  ...
---
### Tokenizer algorithm example
 
- Whitespace Tokenization:
Algorithm: Splits text at spaces, tabs, or newlines.
Example: "This is a sentence." becomes ["This", "is", "a", "sentence."].
- Word Tokenization:
Algorithm: Uses rules to split text into words, often handling punctuation.
Example: "I love pizza!" becomes ["I", "love", "pizza", "!"].
- Character Tokenization:
Algorithm: Splits text into individual characters.
Example: "Hello" becomes ["H", "e", "l", "l", "o"].
- Subword Tokenization (e.g., BPE):
Algorithm: Merges frequent character pairs into new tokens, balancing word and character-level granularity.
Example: "unhappiness" might become ["un", "happi", "ness"], handling the prefix "un-" and suffix "-ness". 
<br>
** Modern Industry Standards ** <br>
As of 2026, subword tokenization algorithms are the industry standard for Large Language Models (LLMs) like GPT-4 because they balance vocabulary size with the ability to handle rare words. 
- Byte Pair Encoding (BPE): Used by GPT models, it iteratively merges the most frequent pairs of characters or tokens.
- WordPiece: Used by BERT, it is similar to BPE but uses a likelihood-based merging strategy.
- SentencePiece: A language-neutral system that treats whitespace as a normal symbol, making it effective for languages like Chinese or Japanese that do not use spaces.
- Unigram: A statistical algorithm that starts with a large vocabulary and prunes it based on the probability of occurrence in a corpus. 

---


### 1. Who creates Dictionary ? (The Engineer + The Algorithm)

The engineer doesn't write the dictionary. Instead, the engineer chooses a **Tokenizer** (a specialized algorithm) and a **Corpus** (a massive pile of text).

* **The Corpus:** This is where "all possible words" come from. For models like **BERT** or **RoBERTa**, the engineers used almost all of English **Wikipedia** and thousands of digital books (**BookCorpus**).


* 
**The Tokenizer:** The engineer runs an algorithm (like *WordPiece* for BERT or *BPE* for RoBERTa) over that massive text. The algorithm counts how often every word appears.


* **The Selection:** The algorithm keeps the top, most frequent words (e.g., the top 30,000) and discards the rest. This final list of 30,000 words/sub-words is your "Dictionary".

### 2. Is it a "Real" Dictionary?

**No.** A real dictionary (like Oxford or Larousse) is too limited for AI.

* **The Problem:** Real dictionaries don't have slang, typos, or new tech words like "ChatGPT" or "TikTok."
* **The AI Solution (Sub-words):** If a word isn't in its 30,000-word list, the AI breaks it into pieces.
* Example: If the dictionary doesn't have **"Yassine"**, it might see it as **"Yass"** + **"ine"**.
* This way, the model can understand *any* word in the world, even if it has never seen it before, by looking at its parts.


### 3. Converting to Vectors: The "Learning" Phase

You asked: *"Use the real dictionary and convert it to vectors?"*
The answer is: **We don't convert them; we let the model "invent" them.**

1. **Initial State:** Every word in that 30,000-word list is assigned a **random vector** (768 random numbers). At this stage, the word "king" and "apple" look exactly the same to the AI—just random noise.
2. **Training:** As you train the model on Wikipedia, it sees "king" near "queen" and "palace" millions of times.
3. **The Optimization:** The model adjusts those 768 numbers so that "king" and "queen" become mathematically similar (their dot product becomes high).
4. **The Result:** The "real" vector isn't found in a book; it is a **learned representation** of how that word is used in human history.

-> Now how do we get from a simple Id like 104 to a vector of 768 numbers? **we use a embedding matrix** : 
size of this matrix : is if our dict has 30K words and our dimension 768 , this matrix is 30000*768.
becouse the model serching in this matrix .

---
### What each number in the vector mean (768) : 
Each number in a 768-dimension vector (embedding) represents a **Feature** or a **Semantic Dimension**.

You should think of these numbers as "coordinates" in a 768-dimensional space where the model maps the meaning of words.

---

### 1. What a "Feature" actually is

If you could "open up" the vector for the word **"king"**, each of those 768 numbers would act as a specific "meaning score":

* **Dimension 1 (Gender):** Might be a high positive number like **0.95**, representing "Masculinity."
* **Dimension 2 (Royalty):** Might be **0.88**, representing "Power/Ruler."
* **Dimension 3 (Food):** Might be **-0.01**, because a king is not typically an object you eat.
* **Dimension 4 (Age):** Might be **0.40**, representing a tendency toward being an "Adult."

### 2. Human vs. AI Understanding

There is a big catch you should mention in your presentation: **We cannot usually name these dimensions.**

* **Human Intuition:** We would love to say "Dimension #45 is definitely the 'Color' dimension."
* **AI Reality:** The model finds its own 768 patterns that work best for math. Dimension #45 might be a weird mix of "Is a plural noun" + "Is related to nature" + "Usually found in the beginning of a sentence." This is called **Distributed Representation**.

### 3. Why 768? (The "Resolution" of Meaning)

Why not just 3 numbers? Because the human language is too complex.

* If you only have 3 numbers, "King," "Prince," and "Duke" would all look exactly the same.
* With 768 numbers, the model has enough "room" to distinguish the tiny differences between a "King" (sovereign) and a "Prince" (heir).
> so feature like paramatere? : <br>

  You are absolutely correct—you’ve hit on the most important concept in modern AI!

Yes, every one of those 768 numbers is a **parameter**. And yes, the model "chooses" (learns) what those numbers should represent to best understand language.

---

### 1. Who creates the "Gender" column?

In the old days of AI, engineers tried to manually create columns:

* Column 1: Gender
* Column 2: Royalty
* Column 3: Is a name?

**But humans are bad at this.** We can't think of 768 different categories for every word in the world.

So, we let the **Model** do it. We give the model 768 "empty slots" (random numbers) for the word "Yassine". During training, the model realizes: *"Wait, every time I see 'Yassine,' I also see the word 'he' or 'man.' I should change the value in Slot #12 to a high number to represent this pattern."*

### 2. The Model as the "Decision Maker"

The model "chooses" the best features through **optimization**.

* It doesn't "know" what gender is in a human sense.
* It only knows that "Yassine," "King," and "Man" all share a similar mathematical pattern in a specific column.
* If the model notices that "Yassine" and "King" always appear in sentences about power, it will create a "Power" column (Parameter) automatically.

### 3. "Yassine" in the Vector vs. the Word

You asked: *"In the vector, we find a column for gender, but for the word, we don't find a gender column?"*

Exactly!

* **The Text Word:** "Yassine" is just 7 letters. It contains no metadata.
* **The Vector (Embedding):** This is where the "hidden meaning" lives. The vector is the model's **translation** of those 7 letters into 768 meaningful numbers (parameters).


---

### RNN : 
![structor of RNN](Screenshot%20from%202026-01-13%2013-18-38.png)
-> how the calculs happen ?

### 1. The Initial Hidden State ($a^{<0>} = \vec{0}$)

 Is typically a vector of zeros.

* **The Reason:** Since "Yassine" is the very first word, the model has no "past" to remember yet. The zeros simply represent an "empty memory".

### 2. The Weights  ($W_{aa}$ and $W_{ax}$)  are the "Brain"

Think of the weights as the model's **knowledge of the English language**.

* **Initialization:** When you start training, these matrices are filled with **small random numbers**, not zeros.
* **Learning:** During training, the model adjusts these numbers so it can understand that if the first word is a name like "Yassine," the next word is likely a verb like "is."

### 3. The Math for the First Word

Let's look at your equation for the first step ():

* **The  part:** Because it multiplies by the zeros of , this part of the equation becomes zero **for this step only**. The "memory" contribution is empty.
* **The  part:** This is where the magic happens!  (which is NOT zero) multiplies the vector for "Yassine" (). This allows the model to capture the meaning of the first word and store it in the new hidden state .

Example : 

Let’s look at the "Yassine" calculation with real numbers. To keep the math simple for your presentation, we will use an embedding size of **3** and a hidden size of **3** (instead of 768).

### 1. The Inputs

* ** (Embedding for "Yassine"):** $[1, 2, 3]^T$.
* $W_{ax}$** (Learned Weights):**
  
        $$\begin{bmatrix} 0.1 & 0.2 & 0.3 \\ 0.4 & 0.5 & 0.6 \\ 0.7 & 0.8 & 0.9 \end{bmatrix}$$
  
### 2. The Step-by-Step Multiplication (($W_{ax} \cdot x^{<1>}$))

This is a **Matrix-Vector Multiplication**, which is essentially **3 separate dot products**.

* **Dot Product 1 (Top row):**
  
        $$(0.1 \times 1) + (0.2 \times 2) + (0.3 \times 3) = 0.1 + 0.4 + 0.9 = \mathbf{1.4}$$

* **Dot Product 2 (Middle row):**

        $$(0.4 \times 1) + (0.5 \times 2) + (0.6 \times 3) = 0.4 + 1.0 + 1.8 = \mathbf{3.2}$$

* **Dot Product 3 (Bottom row):**
  
        $$(0.7 \times 1) + (0.8 \times 2) + (0.9 \times 3) = 0.7 + 1.6 + 2.7 = \mathbf{5.0}$$


### 3. The Result

The output of the $W_{ax}$ part is a new vector:
        
        $$\begin{bmatrix} 1.4 \\ 3.2 \\ 5.0 \end{bmatrix}$$

### What happens next in the formula?

According to the RNN equation:

**Addition:** This result $[1.4, 3.2, 5.0]$ is added to the memory part ($W_{aa} a^{<0>}$) and the bias ($b_a$).2. 
**Activation:** The sum goes into  g (tanh) to squash the numbers between -1 and 1.
3. **Final State:** The result is a—the model's first "memory" of "Yassine".

**Key Takeaway for your slide:**
The matrix  acts as a "translator" that takes the raw word vector and extracts specific features (like "Is this a person?" or "Is this the start of a sentence?") into the model's internal memory space.

-> how we choose b? : 
$b_a$ is a learnable parameter, exactly like the weight matrices $W_{ax}$ and $W_{aa}$.
b is also matrix <br>
so the result is matrice like this 

$$a^{<1>} = \begin{bmatrix} 0.885 \\ 0.997 \\ 1.000 \end{bmatrix}$

ok we get the a lets calcul y : 
- $W_{ya}$ (Learned Weights): $\begin{bmatrix} 0.5 & -0.1 & 0.8 \end{bmatrix}$ (Size $1 \times 3$)$b_y$ 
- (Bias): $[0.1]$
This is the final step of the RNN process—taking the "memory" you just created and turning it into an actual **prediction**.

In your diagrams, the notation **** in the weight matrix  stands for **"from  to "**. This matrix is the "bridge" between the hidden layer and the final output layer.

### 1. The Output Equation

According to your slide, the output  is calculated as:


* ****: This is the hidden state (the "memory") you just calculated.
* ****: This is a learned weight matrix that translates that memory into something the outside world understands (like a word prediction).
* ****: This is the second activation function, usually a **Softmax** if we are trying to choose a word from a dictionary.

---

### 2. Practical Example using our "Yassine" results

Let's use the  vector we just calculated: .

Imagine our model is trying to predict if the sentence is "Positive" or "Negative." Our output  only needs to be **one number** (Probability).

**The Setup:**

* **Input **: 
* ** (Learned Weights)**:  (Size )
* ** (Bias)**: 

**The Math Step-by-Step:**

1. **The Dot Product (($W_{ya} \cdot a^{<1>}$)):**


        $$(0.5 \times 0.885) + (-0.1 \times 0.997) + (0.8 \times 1.000)$$$$0.4425 - 0.0997 + 0.8 = \mathbf{1.1428}$$

2. **Add Bias:**
   
        $$1.1428 + 0.1 = \mathbf{1.2428}$$

4. **Apply  (Sigmoid for probability):**

        $$\text{Sigmoid}(1.2428) \approx \mathbf{0.776}$$


**The Result ():** 0.776. This means the model is **77.6% sure** that after the word "Yassine," the sentence is heading in a positive direction.

---

-> Ok what limtation for RNN ? 
### Limitation : 

1. Vanishing and Exploding Gradients
This remains the most famous drawback. During training, as errors are backpropagated through many time steps (Backpropagation Through Time), the gradients are repeatedly multiplied by weights. 
Vanishing: Gradients become extremely small, approaching zero, which prevents the network from learning long-term dependencies because the weights in earlier layers stop updating.
Exploding: Gradients grow exponentially large, causing numerical instability and making the model's weights diverge. 
2. Sequential Processing Bottleneck
RNNs process data linearly (word by word), meaning the computation for the current step depends on the result of the previous step. 
Lack of Parallelism: Unlike Transformers, which process entire sequences at once, RNNs cannot fully leverage modern parallel hardware like GPUs/TPUs, leading to significantly slower training and inference times for long sequences. 
3. Limited "Long-Term" Memory
Even with variants like LSTMs or GRUs, basic RNNs struggle to maintain information across long gaps. 
Bias Towards Recent Data: The "hidden state" (memory) is a fixed size. As new information arrives, older context is progressively diluted or overwritten, causing the model to "forget" the beginning of a long document by the time it reaches the end. 
4. Difficulty with Global Context
Because RNNs process sequences in a specific order (usually left-to-right), they often fail to capture the "global" context of a sentence where a word's meaning might depend heavily on a word that appears much later.

thier solution solved the vanishing problem is LSTM but also still process data linearly (word by word)

-> That why the engineers create "the attention is all what you need" paper

---

### TRANSFORMER : 
Look at every word simultaneusly . there is no previous state a(t-1). it use Self attention instead of use W(ax),W(aa) .
in Transformer we have new wieght matrices that act a "search engine" for meaning :
- Query(Q) : what a word is looking for.
- Key(K) : what a word contains(like identity)
- Value(V) : the actual information the word provides once a match is found

![Tranformer Architictor](transformers.webp)
![Math](Screenshot%20from%202026-01-13%2014-06-15.png)


Let’s look at the word "king" and see how it interacts with the rest of the sentence.(ex : Yassine is the king of the world) hhhh. <br>
- Query ($Q$): Think of this as a Search Query.For "king": "I am a royal title. Is there a person or a name in this sentence that I belong to?"
- Key ($K$): Think of this as a Label or Identity Card.For "Yassine": "I am a proper noun, a person, and a subject."For "world": "I am a physical location/concept."
- Value ($V$): Think of this as the Information the word carries.For "Yassine": The actual semantic meaning of "Yassine."




