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
