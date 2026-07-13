# 🛡️ SpamShield AI
### AI-Powered Email & SMS Spam Detection System

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)

![Streamlit](https://img.shields.io/badge/Streamlit-Web%20Application-red?style=for-the-badge&logo=streamlit)

![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)

![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-AI-orange?style=for-the-badge&logo=scikitlearn)

![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

</p>

---




# 📌 Project Overview

SpamShield AI is a Machine Learning based Email and SMS Spam Detection System designed to identify spam, phishing, and legitimate messages with high accuracy.

The application uses Natural Language Processing (NLP) techniques combined with a trained Machine Learning model to classify incoming messages into:

- ✅ Legitimate Message
- 🚨 Spam Message

The complete application is developed using **Python** and **Streamlit**, providing a clean, responsive, and modern web interface.

---

# 🚀 Features

### Machine Learning

- Pre-trained Spam Detection Model
- TF-IDF Text Vectorization
- Natural Language Processing
- Text Preprocessing
- Binary Classification

---

# ⚡ Quick Test

1. Open the application.
2. Copy any message from the **Sample Messages** section.
3. Paste it into the text box.
4. Click **Analyze Message**.
5. Verify whether the prediction matches the **Expected Result**.

---

### User Interface

- Modern SaaS Dashboard
- Professional Clean Design
- Responsive Layout
- Interactive Analytics
- Live Prediction
- Message Statistics
- Confidence Score (if supported)
- Session Analytics

---

### Security Features

- Spam Detection
- Phishing Detection
- Malicious Message Identification
- Safe Message Classification

---

# 📂 Project Structure

```
SpamShield-AI/
│
├── app.py
├── model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── .gitignore
├── spam.csv (optional)
└── screenshots/
```

---

# 🧠 Machine Learning Pipeline

```
User Message
      │
      ▼
Text Preprocessing
      │
      ▼
Lowercase Conversion
      │
      ▼
Tokenization
      │
      ▼
Stopword Removal
      │
      ▼
Stemming
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Machine Learning Model
      │
      ▼
Spam / Legitimate
```

---

# ⚙️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| Streamlit | Web Application |
| Scikit-Learn | Machine Learning |
| NLTK | NLP Processing |
| Pandas | Data Handling |
| NumPy | Numerical Computing |
| Pickle | Model Serialization |

---

# 🧠 Machine Learning Workflow

### Data Collection

Spam Dataset

↓

### Data Cleaning

↓

### Text Preprocessing

↓

### Feature Extraction

↓

### TF-IDF Vectorization

↓

### Model Training

↓

### Model Evaluation

↓

### Save Model (.pkl)

↓

### Deploy using Streamlit

---

# 📊 Features Included

- Email Spam Detection
- SMS Spam Detection
- Message Analysis
- Character Counter
- Word Counter
- Sentence Counter
- Link Detection
- Email Detection
- Phone Number Detection
- Response Time
- Session Statistics

---

# 📸 Screenshots

## Home Page

(Add Screenshot Here)

---

## Detection Page

(Add Screenshot Here)

---

## Analytics Dashboard

(Add Screenshot Here)

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/vikashkumarsingh21/SMS-Spam-Classifier.git
```

Go to project folder

```bash
cd SpamShield-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

# 🌐 Live Demo

Website

```
https://spam-shield-ai.streamlit.app/
```

---

# 🧪 Sample Messages for Testing

Use the following sample messages to test the SpamShield AI application.

---

## ✅ Legitimate (Not Spam) Messages

### Sample 1
```
Hi Vikas,

Your interview has been scheduled for Monday at 10:00 AM.
Please bring your ID card and resume.

Thank you,
HR Team
```

---

### Sample 2
```
Hello,

Your food order has been successfully placed.
It will be delivered within 30 minutes.

Thank you for choosing us.
```

---

### Sample 3
```
Hey bro,

Are we meeting in the college library today?
Let me know your timing.

Thanks!
```

---

### Sample 4
```
Dear Customer,

Your electricity bill has been successfully paid.

Transaction ID: 84572931

Thank you.
```

---

### Sample 5
```
Good Morning,

Your Amazon package has been shipped and is expected to arrive tomorrow.

Track your order in the app.
```

---

# 🚨 Spam Messages

### Sample 1
```
Congratulations!!

You have won ₹50,000 Cash Prize.

Click the link below immediately to claim your reward.

http://claim-now-free.xyz
```

---

### Sample 2
```
URGENT!!

Your bank account will be suspended.

Verify your account now.

https://secure-bank-login-free.com
```

---

### Sample 3
```
WIN an iPhone 16 Pro FREE!!

Limited Offer.

Click here now.

www.freeiphoneoffer.xyz
```

---

### Sample 4
```
Congratulations!

You have been selected for a FREE vacation package.

Call now: +91 9876543210

Offer expires today.
```

---

### Sample 5
```
You have received a lottery prize of ₹25 Lakhs.

Send your bank details immediately to receive the money.

Limited Time Offer.
```

---

### Sample 6
```
Exclusive Offer!!

Get 90% Discount on all products.

Visit

www.super-sale-now.xyz

Offer ends in 10 minutes.
```

---

### Sample 7
```
Dear User,

Your Paytm account has been blocked.

Click below to verify your KYC.

http://paytm-secure-update.xyz
```

---

### Sample 8
```
Congratulations!!

You are our lucky winner.

Claim your FREE Samsung Galaxy S25 Ultra today.

Click here now.
```

---

## ⚠️ Note

The prediction depends on the Machine Learning model used during training.

Different models and datasets may classify some messages differently based on the learned patterns.

---

# 📈 Model Information

| Property | Value |
|----------|-------|
| Task | Spam Detection |
| Input | Email / SMS |
| Output | Spam or Legitimate |
| NLP | NLTK |
| Feature Extraction | TF-IDF |
| Deployment | Streamlit |

---

# 📦 Dependencies

- Python 3.10+
- Streamlit
- Scikit-Learn
- NLTK
- NumPy
- Pandas
- Joblib

---

# 📚 Future Improvements

- Multi-language Spam Detection
- Real-time Email Scanner
- Gmail Integration
- SMS API Integration
- Deep Learning Model
- Transformer Models (BERT)
- Mobile Application
- User Authentication
- Cloud Database
- Admin Dashboard

---

# 🎯 Applications

- Email Security
- SMS Security
- Cybersecurity
- Enterprise Communication
- Banking
- Education
- Government Organizations
- Personal Use

---

# 🔒 Security

This project performs all predictions locally using the trained Machine Learning model.

No user messages are permanently stored or shared with third-party services.

---

# 🤝 Contribution

Contributions are welcome.

If you have suggestions or improvements, feel free to open an Issue or submit a Pull Request.

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Developer

**Vikas Kumar**

B.Tech Computer Science Engineering (Artificial Intelligence & Machine Learning)

Machine Learning Enthusiast | Python Developer | AI Developer

GitHub:
https://github.com/vikashkumarsingh21/

LinkedIn:
https://www.linkedin.com/in/vikas-kumar-0803r/

Email:
vk0102103@email.com

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.

It helps others discover the project and supports future development.

---

<p align="center">

Made with ❤️ using Python, Streamlit and Machine Learning

</p>