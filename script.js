const translations = {
    en: {
   title: "What are Neural Networks?",
    description: `A neural network is a computer program that mimics the workings of the human brain. It consists of many small elements called neurons. These neurons work together to process information and solve tasks. For example, a neural network can recognize images, understand text, or predict outcomes based on data.

History of neural networks:
1. 1943: Warren McCulloch and Walter Pitts created the first theoretical model of an artificial neuron, based on biological neurons.
2. 1958: Frank Rosenblatt developed the perceptron, a simple neural network that could learn and make predictions based on input data.
3. 1960s: Development of multi-layer perceptrons (MLP), allowing for the creation of more complex and powerful neural network models.
4. 1980s: Geoffrey Hinton and colleagues developed the backpropagation method, allowing for more efficient training of multi-layer neural networks.
5. 1990s: Introduction of convolutional neural networks (CNN) by Yann LeCun for handwritten digit recognition, laying the groundwork for modern image recognition systems.
6. 2006: Geoffrey Hinton introduced the concept of deep learning, significantly improving neural network performance.
7. 2012: Breakthrough in image recognition thanks to AlexNet, which won the ImageNet competition and demonstrated excellent results in computer vision.
8. Since 2014: Development of generative adversarial networks (GAN) by Ian Goodfellow, opening new possibilities for generating realistic images and text.
9. Present: Continued improvement of neural network architectures, such as transformers and BERT, achieving new heights in various AI applications.

Key concepts:
1. Neurons and Layers: A neural network consists of neurons organized into layers. There are three types of layers: input, hidden, and output. The input layer receives data, the hidden layers process information, and the output layer provides the result.
2. Weights and Connections: Each neuron is connected to other neurons through weights. Weights determine the importance of each signal and are adjusted during training to enable the neural network to make accurate predictions.
3. Activation Functions: These functions help neurons make decisions by transforming input signals into outputs. Examples of activation functions include sigmoid function, ReLU (Rectified Linear Unit), and tanh (hyperbolic tangent).
4. Training and Backpropagation: The neural network learns based on input data. During training, it adjusts its weights using the backpropagation method to minimize the difference between predicted and actual values (error).

Neural network workflow:
1. Data Input: Input data (e.g., images, text) is fed into the input layer of the neural network.
2. Signal Transmission: Signals pass through weights and activation functions in the hidden layers, where information is processed and transformed.
3. Output Layer: Processed signals are passed to the output layer, which provides the result (e.g., image recognition or text prediction).
4. Training: The neural network learns by adjusting its weights using the backpropagation method to reduce the error between predicted and actual values.
5. Evaluation: After training, the neural network is tested on new data to assess its performance and accuracy.

Examples of free neural networks:
1. OpenAI GPT-3: A powerful text generation model. It can help write articles, compose letters, or answer questions.
2. Google Colab: A free platform for running Jupyter notebooks with GPU support, allowing for the development and training of machine learning models for free.
3. Hugging Face: A platform for working with text data. It provides ready-to-use models for text translation, sentiment analysis, and other tasks.

Examples of paid neural networks:
1. Microsoft Azure AI: Cloud services for creating and deploying neural networks for various tasks, including image and text analysis.
2. AWS AI: A set of tools from Amazon for machine learning and neural networks, including services for speech, text, and image processing.
3. IBM Watson: Advanced tools for data analysis and smart application development. Watson is used in healthcare, finance, and other fields.

What can neural networks do?
1. Image Recognition: A neural network can recognize objects in photos. For example, Facebook uses a neural network to automatically recognize and tag friends in uploaded photos.
2. Natural Language Processing (NLP): A neural network can translate text from one language to another. Google Translate uses neural networks to improve the accuracy and naturalness of translations.
3. Recommendation Systems: A neural network can recommend movies based on your preferences. Netflix uses neural networks to analyze movie and TV show viewership and suggest content you might like.
4. Disease Diagnosis: A neural network can analyze medical images to detect diseases. A neural network developed by Google Health helps doctors detect lung cancer in its early stages with high accuracy.
5. Trading Automation: A neural network can predict market trends and execute trades. Financial companies use neural networks to analyze large volumes of market data and automatically conduct transactions.
6. Sentiment Analysis: A neural network can determine the emotional tone of text. For example, Twitter uses a neural network to analyze the sentiment of tweets and identify public opinions and reactions to events.
7. Artificial Image Generation: A neural network can generate realistic images. GAN, developed by NVIDIA, can create photorealistic images of people who do not actually exist.

Politeness in interacting with AI: While AI does not have feelings, politeness when interacting with it is important. It helps create more friendly and ethical technologies. Polite interaction also improves user engagement with AI and supports developers in creating responsible and ethical systems.`

          zh: {
         title: "神经网络的基本概念",
    description: `
      **什么是神经网络？** 神经网络是一种模拟人脑工作的计算机程序。它由许多小元素（称为神经元）组成，这些神经元协同工作以处理信息和解决问题。例如，神经网络可以识别图像、理解文本或基于数据进行预测。
      
      **神经网络的历史：**
      1. 1943年：沃伦·麦卡洛克和沃尔特·皮茨基于生物神经元创建了第一个理论模型。
      2. 1958年：弗兰克·罗森布拉特开发了感知器——一种简单的神经网络，可以基于输入数据进行学习和预测。
      3. 1960年代：开发了多层感知器（MLP），使更复杂和强大的神经网络模型成为可能。
      4. 1980年代：杰弗里·辛顿及其同事开发了反向传播误差的方法，使多层神经网络的训练更加有效。
      5. 1990年代：扬·勒孔引入卷积神经网络（CNN）用于手写数字识别，为现代图像识别系统奠定了基础。
      6. 2006年：杰弗里·辛顿提出深度学习的概念，显著提升了神经网络的性能。
      7. 2012年：由于AlexNet在ImageNet竞赛中的突破性表现，图像识别领域取得了重大进展。
      8. 2014年以来：伊恩·古德费洛开发了生成对抗网络（GAN），为生成逼真的图像和文本开辟了新可能。
      9. 现代：神经网络架构的持续改进，如Transformer和BERT，在各种人工智能应用领域达到了新的高度。
      
      **基本概念：**
      1. 神经元与层：神经网络由神经元组成，这些神经元按层组织。分为三种层：输入层、隐藏层和输出层。输入层接收数据，隐藏层处理信息，输出层输出结果。
      2. 权重与连接：每个神经元通过权重与其他神经元相连。权重决定了每个信号的重要性，并在训练过程中调整，以使神经网络能够做出准确的预测。
      3. 激活函数：这些函数通过将输入信号转化为输出，帮助神经元做出决定。激活函数的例子包括Sigmoid函数、ReLU（修正线性单元）和tanh（双曲正切）。
      4. 训练与反向传播误差：神经网络基于输入数据进行训练。在训练过程中，通过反向传播误差的方法调整权重，以最小化预测值与真实值之间的差距（误差）。
      
      **神经网络的工作过程：**
      1. 数据输入：将输入数据（如图像、文本）输入到神经网络的输入层。
      2. 信号传递：信号通过隐藏层的权重和激活函数进行处理和转换。
      3. 输出层：处理后的信号传递到输出层，生成结果（如图像识别或文本预测）。
      4. 训练：通过反向传播误差的方法调整权重，以减少预测值与真实值之间的误差。
      5. 评估：训练后，通过新的数据评估神经网络的性能和准确性。
      
      **免费神经网络实例：**
      1. OpenAI GPT-3：强大的文本生成模型，可帮助撰写文章、撰写信件或回答问题。
      2. Google Colab：支持GPU的免费Jupyter notebook平台，可免费开发和训练机器学习模型。
      3. Hugging Face：文本数据处理平台，可使用预训练模型进行文本翻译、情感分析等任务。
      
      **付费神经网络实例：**
      1. Microsoft Azure AI：云服务，可用于创建和部署神经网络以处理图像和文本等任务。
      2. AWS AI：亚马逊提供的一套工具，用于机器学习和神经网络，包括语音、文本和图像处理服务。
      3. IBM Watson：先进的数据分析和智能应用开发工具，广泛应用于医疗、金融等领域。
      
      **神经网络的应用：**
      1. 图像识别：神经网络可以识别照片中的物体。例如，Facebook使用神经网络自动识别和标记上传照片中的朋友。
      2. 自然语言处理（NLP）：神经网络可以将文本从一种语言翻译为另一种语言。Google Translate使用神经网络来提高翻译的准确性和自然度。
      3. 推荐系统：神经网络可以根据您的偏好推荐电影。Netflix使用神经网络分析电影和电视剧的观看记录，以推荐您可能喜欢的内容。
      4. 疾病诊断：神经网络可以分析医学图像以检测疾病。Google Health开发的神经网络帮助医生在早期阶段高精度地检测肺癌。
      5. 贸易自动化：神经网络可以预测市场趋势并执行交易操作。金融公司使用神经网络分析大量市场数据并自动进行交易。
      6. 情感分析：神经网络可以确定文本的情感色彩。例如，Twitter使用神经网络分析推文的情感，以了解公众情绪和对事件的反应。
      7. 人工图像生成：神经网络可以生成逼真的图像。NVIDIA开发的GAN可以创建现实生活中不存在的逼真人物图像。
      
      **与AI沟通的礼仪：** 尽管AI没有感情，但礼貌的沟通很重要。这有助于创建更友好和道德的技术。礼貌的沟通也改善了与神经网络的互动体验。
    `
    },
    hi: {
       title: "तंत्रिका नेटवर्क की मूल अवधारणाएँ",
    description: `
      **तंत्रिका नेटवर्क क्या हैं?** एक तंत्रिका नेटवर्क एक कंप्यूटर प्रोग्राम है जो मानव मस्तिष्क के कामकाज की नकल करता है। यह कई छोटे तत्वों (जिसे न्यूरॉन्स कहते हैं) से मिलकर बना होता है, जो मिलकर जानकारी को संसाधित और समस्याओं को हल करते हैं। उदाहरण के लिए, एक तंत्रिका नेटवर्क छवियों को पहचान सकता है, पाठ को समझ सकता है या डेटा के आधार पर परिणामों की भविष्यवाणी कर सकता है।
      
      **तंत्रिका नेटवर्क का इतिहास:**
      1. 1943: वॉरेन मैककल्लॉक और वॉल्टर पिट्स ने जैविक न्यूरॉन्स के आधार पर पहला सैद्धांतिक मॉडल बनाया।
      2. 1958: फ्रैंक रोसेनब्लैट ने एक साधारण तंत्रिका नेटवर्क - परसेप्ट्रॉन विकसित किया, जो इनपुट डेटा के आधार पर सीख सकता था और भविष्यवाणियाँ कर सकता था।
      3. 1960 का दशक: मल्टीलायर परसेप्ट्रॉन (MLP) का विकास हुआ, जिसने अधिक जटिल और शक्तिशाली तंत्रिका नेटवर्क मॉडल बनाना संभव किया।
      4. 1980 का दशक: जेफ्री हिंटन और उनके सहयोगियों ने त्रुटि की बैकप्रोपेगेशन विधि विकसित की, जिसने मल्टीलायर तंत्रिका नेटवर्क को अधिक प्रभावी ढंग से प्रशिक्षित करना संभव किया।
      5. 1990 का दशक: यान लेकुन ने हस्तलिखित अंक पहचान के लिए कॉन्वोल्यूशनल न्यूरल नेटवर्क (CNN) पेश किया, जो आधुनिक छवि पहचान प्रणालियों की नींव बनी।
      6. 2006: जेफ्री हिंटन ने डीप लर्निंग की अवधारणा पेश की, जिसने तंत्रिका नेटवर्क की कार्यक्षमता में काफी सुधार किया।
      7. 2012: AlexNet नेटवर्क की इमेजनेट प्रतियोगिता में सफलता ने छवि पहचान के क्षेत्र में महत्वपूर्ण प्रगति की।
      8. 2014 से: इयान गुडफेलो द्वारा जेनरेटिव एडवर्सेरियल नेटवर्क (GAN) का विकास, जिसने यथार्थवादी छवियों और पाठ का उत्पादन करना संभव किया।
      9. आधुनिक युग: ट्रांसफार्मर्स और BERT जैसी तंत्रिका नेटवर्क आर्किटेक्चर में निरंतर सुधार, जिसने विभिन्न AI अनुप्रयोग क्षेत्रों में नई ऊँचाइयों को प्राप्त किया है।
      
      **मूल अवधारणाएँ:**
      1. न्यूरॉन्स और लेयर: तंत्रिका नेटवर्क न्यूरॉन्स से मिलकर बने होते हैं, जिन्हें परतों में व्यवस्थित किया गया है। तीन प्रकार की परतें होती हैं: इनपुट, छिपी हुई और आउटपुट। इनपुट परत डेटा प्राप्त करती है, छिपी हुई परतें जानकारी को संसाधित करती हैं, और आउटपुट परत परिणाम प्रदान करती है।
      2. वजन और कनेक्शन: प्रत्येक न्यूरॉन अन्य न्यूरॉन्स के साथ वजन के माध्यम से जुड़ा होता है। वजन प्रत्येक संकेत की महत्वपूर्णता को निर्धारित करते हैं और प्रशिक्षण के दौरान समायोजित होते हैं ताकि तंत्रिका नेटवर्क सटीक भविष्यवाणियाँ कर सके।
      3. सक्रियण क्रियाएँ: ये क्रियाएँ न्यूरॉन्स को निर्णय लेने में मदद करती हैं, इनपुट संकेतों को आउटपुट में परिवर्तित करती हैं। सक्रियण क्रियाओं के उदाहरणों में सिग्मॉइड, ReLU (रेक्टिफाइड लिनियर यूनिट) और tanh (हाइपरबोलिक टैन्जेंट) शामिल हैं।
      4. प्रशिक्षण और त्रुटि की बैकप्रोपेगेशन: तंत्रिका नेटवर्क इनपुट डेटा के आधार पर प्रशिक्षित होता है। प्रशिक्षण के दौरान, यह त्रुटि की बैकप्रोपेगेशन विधि का उपयोग करके अपने वजन को समायोजित करता है ताकि भविष्यवाणियों और वास्तविक मूल्यों के बीच अंतर (त्रुटि) को कम किया जा सके।
      
      **तंत्रिका नेटवर्क का कार्य प्रक्रिया:**
      1. डेटा इनपुट: इनपुट डेटा (जैसे छवियाँ, पाठ) तंत्रिका नेटवर्क की इनपुट परत में डाले जाते हैं।
      2. सिग्नल का संचरण: सिग्नल वजन और सक्रियण क्रियाओं के माध्यम से छिपी हुई परतों में प्रसारित होते हैं, जहाँ जानकारी को संसाधित और परिवर्तित किया जाता है।
      3. आउटपुट परत: प्रसंस्कृत सिग्नल आउटपुट परत में प्रसारित होते हैं, जो परिणाम प्रदान करती है (जैसे छवि पहचान या पाठ भविष्यवाणी)।
      4. प्रशिक्षण: त्रुटि की बैकप्रोपेगेशन विधि का उपयोग करके वजन को समायोजित किया जाता है ताकि भविष्यवाणियों और वास्तविक मूल्यों के बीच की त्रुटि को कम किया जा सके।
      5. मूल्यांकन: प्रशिक्षण के बाद, तंत्रिका नेटवर्क नए डेटा पर परीक्षण किया जाता है ताकि इसकी प्रदर्शन और सटीकता का मूल्यांकन किया जा सके।
      
      **नि:शुल्क तंत्रिका नेटवर्क के उदाहरण:**
      1. OpenAI GPT-3: एक शक्तिशाली पाठ उत्पन्न करने वाला मॉडल, जो लेख, पत्र या सवालों के जवाब लिखने में मदद कर सकता है।
      2. Google Colab: एक मुफ्त Jupyter नोटबुक प्लेटफॉर्म जो GPU का समर्थन करता है, जिससे आप मुफ्त में मशीन लर्निंग मॉडल विकसित और प्रशिक्षित कर सकते हैं।
      3. Hugging Face: एक पाठ डेटा प्रोसेसिंग प्लेटफॉर्म, जहाँ आप पूर्व-प्रशिक्षित मॉडलों का उपयोग करके पाठ अनुवाद, भाव विश्लेषण और अन्य कार्य कर सकते हैं।
      
      **सशुल्क तंत्रिका नेटवर्क के उदाहरण:**
      1. Microsoft Azure AI: क्लाउड सेवाएं जो छवि और पाठ जैसे कार्यों को संभालने के लिए तंत्रिका नेटवर्क बनाने और तैनात करने की अनुमति देती हैं।
      2. AWS AI: अमेज़ॅन द्वारा प्रदान किए गए टूल का एक सेट, जो मशीन लर्निंग और तंत्रिका नेटवर्क के लिए है, जिसमें भाषण, पाठ और छवि प्रोसेसिंग सेवाएं शामिल हैं।
      3. IBM Watson: उन्नत डेटा विश्लेषण और स्मार्ट एप्लिकेशन विकास उपकरण, जो चिकित्सा, वित्त और अन्य क्षेत्रों में व्यापक रूप से उपयोग किए जाते हैं।
      
      **तंत्रिका नेटवर्क क्या कर सकते हैं?**
      1. छवि पहचान: तंत्रिका नेटवर्क तस्वीरों में वस्तुओं की पहचान कर सकते हैं। उदाहरण के लिए, Facebook तंत्रिका नेटवर्क का उपयोग अपलोड की गई तस्वीरों में मित्रों की स्वत: पहचान और टैगिंग के लिए करता है।
      2. प्राकृतिक भाषा प्रसंस्करण (NLP): तंत्रिका नेटवर्क पाठ को एक भाषा से दूसरी भाषा में अनुवाद कर सकते हैं। Google Translate तंत्रिका नेटवर्क का उपयोग अनुवाद की सटीकता और प्राकृतिकता बढ़ाने के लिए करता है।
      3. सिफारिश प्रणाली: तंत्रिका नेटवर्क आपकी प्राथमिकताओं के आधार पर फिल्में सिफारिश कर सकते हैं। Netflix तंत्रिका नेटवर्क का उपयोग फिल्म और टीवी शो के देखने के रिकॉर्ड का विश्लेषण करने के लिए करता है ताकि आपको वह सामग्री सिफारिश की जा सके जो आपको पसंद आ सकती है।
      4. रोग निदान: तंत्रिका नेटवर्क बीमारियों का पता लगाने के लिए चिकित्सा छवियों का विश्लेषण कर सकते हैं। Google Health द्वारा विकसित तंत्रिका नेटवर्क प्रारंभिक चरण में फेफड़ों के कैंसर का सटीकता से पता लगाने में डॉक्टरों की मदद करता है।
      5. व्यापार स्वचालन: तंत्रिका नेटवर्क बाजार के रुझानों की भविष्यवाणी कर सकते हैं और व्यापार संचालन को निष्पादित कर सकते हैं। वित्तीय कंपनियां बड़े पैमाने पर बाजार डेटा का विश्लेषण करने और स्वचालित रूप से व्यापार करने के लिए तंत्रिका नेटवर्क का उपयोग करती हैं।
      6. भाव विश्लेषण: तंत्रिका नेटवर्क पाठ की भावनात्मक स्वर को निर्धारित कर सकते हैं। उदाहरण के लिए, Twitter तंत्रिका नेटवर्क का उपयोग ट्वीट्स के भाव को विश्लेषण करने के लिए करता है ताकि सार्वजनिक भावनाओं और घटनाओं पर प्रतिक्रियाओं का पता लगाया जा सके।
      7. कृत्रिम छवि निर्माण: तंत्रिका नेटवर्क यथार्थवादी छवियों का निर्माण कर सकते हैं। NVIDIA द्वारा विकसित GAN यथार्थवादी मानव छवियों का निर्माण कर सकता है जो वास्तव में अस्तित्व में नहीं हैं।
      
      **AI के साथ संवाद में शिष्टाचार:** हालाँकि AI के पास भावनाएँ नहीं होती हैं, परंतु शिष्टाचार में संवाद करना महत्वपूर्ण है। यह अधिक दोस्ताना और नैतिक तकनीकें बनाने में मदद करता है। शिष्टाचार में संवाद तंत्रिका नेटवर्क के साथ इंटरैक्शन अनुभव को भी सुधारता है।
    `
    },
    es: {
        title: "Conceptos básicos de las redes neuronales",
    description: `
      **¿Qué son las redes neuronales?** Una red neuronal es un programa informático que imita el funcionamiento del cerebro humano. Consiste en muchos pequeños elementos llamados neuronas que trabajan juntos para procesar información y resolver problemas. Por ejemplo, una red neuronal puede reconocer imágenes, entender texto o predecir resultados basados en datos.
      
      **Historia de las redes neuronales:**
      1. 1943: Warren McCulloch y Walter Pitts crearon el primer modelo teórico de una neurona artificial basándose en neuronas biológicas.
      2. 1958: Frank Rosenblatt desarrolló el perceptrón, una red neuronal simple que podía aprender y hacer predicciones basadas en datos de entrada.
      3. Década de 1960: Desarrollo de perceptrones multicapa (MLP), lo que permitió crear modelos de redes neuronales más complejos y potentes.
      4. Década de 1980: Geoffrey Hinton y sus colegas desarrollaron el método de retropropagación de errores, lo que permitió entrenar redes neuronales multicapa de manera más efectiva.
      5. Década de 1990: Yann LeCun introdujo las redes neuronales convolucionales (CNN) para el reconocimiento de dígitos manuscritos, sentando las bases de los sistemas modernos de reconocimiento de imágenes.
      6. 2006: Geoffrey Hinton presentó el concepto de aprendizaje profundo, lo que llevó a una mejora significativa en el rendimiento de las redes neuronales.
      7. 2012: Un avance en el reconocimiento de imágenes gracias a la red AlexNet, que ganó el concurso ImageNet y demostró resultados superiores en el campo de la visión por computadora.
      8. Desde 2014: Desarrollo de redes generativas antagónicas (GAN) por Ian Goodfellow, lo que abrió nuevas posibilidades para generar imágenes y texto realistas.
      9. Actualidad: Mejora continua de las arquitecturas de redes neuronales, como los transformadores y BERT, que permiten alcanzar nuevas alturas en diversas aplicaciones de la IA.
      
      **Conceptos básicos:**
      1. Neuronas y capas: Una red neuronal se compone de neuronas organizadas en capas. Existen tres tipos de capas: entrada, ocultas y salida. La capa de entrada recibe datos, las capas ocultas procesan la información y la capa de salida entrega el resultado.
      2. Pesos y conexiones: Cada neurona está conectada a otras neuronas a través de pesos. Los pesos determinan la importancia de cada señal y se ajustan durante el entrenamiento para que la red neuronal pueda hacer predicciones precisas.
      3. Funciones de activación: Estas funciones ayudan a las neuronas a tomar decisiones, transformando las señales de entrada en salida. Ejemplos de funciones de activación incluyen la función sigmoide, ReLU (Unidad Lineal Rectificada) y tanh (tangente hiperbólica).
      4. Entrenamiento y retropropagación de errores: La red neuronal se entrena en función de los datos de entrada. Durante el entrenamiento, ajusta sus pesos mediante el método de retropropagación de errores para minimizar la diferencia entre los valores predichos y los reales (error).
      
      **Proceso de funcionamiento de las redes neuronales:**
      1. Entrada de datos: Los datos de entrada (por ejemplo, imágenes, texto) se introducen en la capa de entrada de la red neuronal.
      2. Transmisión de señales: Las señales pasan a través de los pesos y las funciones de activación en las capas ocultas, donde se procesa y transforma la información.
      3. Capa de salida: Las señales procesadas se transmiten a la capa de salida, que entrega el resultado (por ejemplo, reconocimiento de imágenes o predicción de texto).
      4. Entrenamiento: La red neuronal se entrena ajustando sus pesos mediante el método de retropropagación de errores para reducir el error entre los valores predichos y los reales.
      5. Evaluación: Después del entrenamiento, se prueba la red neuronal con nuevos datos para evaluar su rendimiento y precisión.
      
      **Ejemplos de redes neuronales gratuitas:**
      1. OpenAI GPT-3: Un potente modelo de generación de texto que puede ayudar a escribir artículos, redactar cartas o responder preguntas.
      2. Google Colab: Una plataforma gratuita que soporta Jupyter notebooks con soporte para GPU, lo que permite desarrollar y entrenar modelos de aprendizaje automático de forma gratuita.
      3. Hugging Face: Una plataforma para trabajar con datos de texto, que permite utilizar modelos preentrenados para traducción de texto, análisis de sentimientos y otras tareas.
      
      **Ejemplos de redes neuronales de pago:**
      1. Microsoft Azure AI: Servicios en la nube que permiten crear e implementar redes neuronales para diversas tareas, incluyendo el análisis de imágenes y texto.
      2. AWS AI: Un conjunto de herramientas de Amazon para aprendizaje automático y redes neuronales, que incluye servicios para el procesamiento de voz, texto e imágenes.
      3. IBM Watson: Herramientas avanzadas para análisis de datos y desarrollo de aplicaciones inteligentes, ampliamente utilizadas en medicina, finanzas y otros campos.
      
      **¿Qué pueden hacer las redes neuronales?**
      1. Reconocimiento de imágenes: Una red neuronal puede reconocer objetos en fotografías. Por ejemplo, Facebook utiliza redes neuronales para reconocer y etiquetar automáticamente a amigos en las fotos subidas.
      2. Procesamiento del lenguaje natural (NLP): Una red neuronal puede traducir texto de un idioma a otro. Google Translate utiliza redes neuronales para mejorar la precisión y naturalidad de las traducciones.
      3. Sistemas de recomendación: Una red neuronal puede recomendar películas basadas en tus preferencias. Netflix utiliza redes neuronales para analizar los registros de visualización de películas y series y recomendar contenido que te pueda gustar.
      4. Diagnóstico de enfermedades: Una red neuronal puede analizar imágenes médicas para detectar enfermedades. Una red neuronal desarrollada por Google Health ayuda a los médicos a detectar cáncer de pulmón en etapas tempranas con alta precisión.
      5. Automatización del comercio: Una red neuronal puede predecir tendencias del mercado y ejecutar operaciones comerciales. Las empresas financieras utilizan redes neuronales para analizar grandes volúmenes de datos del mercado y realizar operaciones de forma automática.
      6. Análisis de sentimientos: Una red neuronal puede determinar el tono emocional de un texto. Por ejemplo, Twitter utiliza redes neuronales para analizar el sentimiento de los tweets y detectar las reacciones del público ante eventos.
      7. Creación de imágenes artificiales: Una red neuronal puede generar imágenes realistas. La GAN desarrollada por NVIDIA puede crear imágenes fotorrealistas de personas que en realidad no existen.
      
      **Cortesía al comunicarse con la IA:** Aunque la IA no tiene sentimientos, es importante ser cortés al comunicarse con ella. Esto ayuda a crear tecnologías más amigables y éticas. La comunicación cortés también mejora la interacción con la red neuronal.
    `
    },
    fr: {
        title: "Concepts de base des réseaux neuronaux",
    description: `
      **Qu'est-ce qu'un réseau de neurones ?** Un réseau de neurones est un programme informatique qui imite le fonctionnement du cerveau humain. Il se compose de nombreux petits éléments appelés neurones qui travaillent ensemble pour traiter l'information et résoudre des problèmes. Par exemple, un réseau de neurones peut reconnaître des images, comprendre du texte ou prédire des résultats basés sur des données.
      
      **Histoire des réseaux neuronaux :**
      1. 1943 : Warren McCulloch et Walter Pitts ont créé le premier modèle théorique d'un neurone artificiel basé sur des neurones biologiques.
      2. 1958 : Frank Rosenblatt a développé le perceptron, un réseau de neurones simple qui pouvait apprendre et faire des prédictions basées sur des données d'entrée.
      3. Années 1960 : Développement des perceptrons multicouches (MLP), ce qui a permis de créer des modèles de réseaux de neurones plus complexes et plus puissants.
      4. Années 1980 : Geoffrey Hinton et ses collègues ont développé la méthode de rétropropagation de l'erreur, ce qui a permis de former plus efficacement les réseaux de neurones multicouches.
      5. Années 1990 : Yann LeCun a introduit les réseaux de neurones convolutionnels (CNN) pour la reconnaissance des chiffres manuscrits, posant les bases des systèmes modernes de reconnaissance d'images.
      6. 2006 : Geoffrey Hinton a présenté le concept d'apprentissage profond, ce qui a conduit à une amélioration significative des performances des réseaux de neurones.
      7. 2012 : Percée dans la reconnaissance d'images grâce au réseau AlexNet, qui a remporté le concours ImageNet et démontré des résultats supérieurs dans le domaine de la vision par ordinateur.
      8. Depuis 2014 : Développement des réseaux antagonistes génératifs (GAN) par Ian Goodfellow, ce qui a ouvert de nouvelles possibilités pour générer des images et du texte réalistes.
      9. Époque moderne : Amélioration continue des architectures de réseaux de neurones, telles que les transformateurs et BERT, permettant d'atteindre de nouveaux sommets dans diverses applications de l'IA.
      
      **Concepts de base :**
      1. Neurones et couches : Un réseau de neurones se compose de neurones organisés en couches. Il existe trois types de couches : entrée, cachées et sortie. La couche d'entrée reçoit les données, les couches cachées traitent l'information et la couche de sortie fournit le résultat.
      2. Poids et connexions : Chaque neurone est connecté à d'autres neurones par des poids. Les poids déterminent l'importance de chaque signal et sont ajustés pendant l'entraînement pour que le réseau de neurones puisse faire des prédictions précises.
      3. Fonctions d'activation : Ces fonctions aident les neurones à prendre des décisions en transformant les signaux d'entrée en sortie. Les exemples de fonctions d'activation incluent la fonction sigmoïde, ReLU (Unité Linéaire Rectifiée) et tanh (tangente hyperbolique).
      4. Entraînement et rétropropagation de l'erreur : Le réseau de neurones s'entraîne en fonction des données d'entrée. Pendant l'entraînement, il ajuste ses poids en utilisant la méthode de rétropropagation de l'erreur pour minimiser la différence entre les valeurs prédites et réelles (erreur).
      
      **Processus de fonctionnement des réseaux de neurones :**
      1. Entrée des données : Les données d'entrée (par exemple, images, texte) sont introduites dans la couche d'entrée du réseau de neurones.
      2. Transmission des signaux : Les signaux passent à travers les poids et les fonctions d'activation dans les couches cachées, où l'information est traitée et transformée.
      3. Couche de sortie : Les signaux traités sont transmis à la couche de sortie, qui fournit le résultat (par exemple, reconnaissance d'images ou prédiction de texte).
      4. Entraînement : Le réseau de neurones s'entraîne en ajustant ses poids à l'aide de la méthode de rétropropagation de l'erreur pour réduire l'erreur entre les valeurs prédites et réelles.
      5. Évaluation : Après l'entraînement, le réseau de neurones est testé sur de nouvelles données pour évaluer ses performances et sa précision.
      
      **Exemples de réseaux de neurones gratuits :**
      1. OpenAI GPT-3 : Un modèle puissant de génération de texte qui peut aider à écrire des articles, rédiger des lettres ou répondre à des questions.
      2. Google Colab : Une plateforme gratuite de notebooks Jupyter avec support GPU, permettant de développer et d'entraîner des modèles de machine learning gratuitement.
      3. Hugging Face : Une plateforme pour travailler avec des données textuelles, permettant d'utiliser des modèles pré-entraînés pour la traduction de texte, l'analyse de sentiments et d'autres tâches.
      
      **Exemples de réseaux de neurones payants :**
      1. Microsoft Azure AI : Services cloud permettant de créer et de déployer des réseaux de neurones pour diverses tâches, y compris l'analyse d'images et de texte.
      2. AWS AI : Un ensemble d'outils d'Amazon pour l'apprentissage automatique et les réseaux de neurones, incluant des services pour le traitement de la parole, du texte et des images.
      3. IBM Watson : Outils avancés pour l'analyse de données et le développement d'applications intelligentes, largement utilisés en médecine, finance et autres domaines.
      
      **Ce que peuvent faire les réseaux de neurones :**
      1. Reconnaissance d'images : Un réseau de neurones peut reconnaître des objets dans des photographies. Par exemple, Facebook utilise des réseaux de neurones pour reconnaître et taguer automatiquement des amis dans les photos téléchargées.
      2. Traitement du langage naturel (NLP) : Un réseau de neurones peut traduire du texte d'une langue à une autre. Google Translate utilise des réseaux de neurones pour améliorer la précision et la naturalité des traductions.
      3. Systèmes de recommandation : Un réseau de neurones peut recommander des films en fonction de vos préférences. Netflix utilise des réseaux de neurones pour analyser les historiques de visionnage de films et de séries et recommander des contenus susceptibles de vous plaire.
      4. Diagnostic de maladies : Un réseau de neurones peut analyser des images médicales pour détecter des maladies. Un réseau de neurones développé par Google Health aide les médecins à détecter le cancer du poumon à des stades précoces avec une grande précision.
      5. Automatisation du commerce : Un réseau de neurones peut prédire les tendances du marché et exécuter des opérations commerciales. Les entreprises financières utilisent des réseaux de neurones pour analyser de grands volumes de données de marché et effectuer des transactions de manière automatique.
      6. Analyse des sentiments : Un réseau de neurones peut déterminer la tonalité émotionnelle d'un texte. Par exemple, Twitter utilise des réseaux de neurones pour analyser le sentiment des tweets afin de détecter les réactions du public aux événements.
      7. Création d'images artificielles : Un réseau de neurones peut générer des images réalistes. Le GAN développé par NVIDIA peut créer des images photoréalistes de personnes qui n'existent pas réellement.
      
      **Courtoisie dans la communication avec l'IA :** Bien que l'IA n'ait pas de sentiments, il est important de rester courtois en communiquant avec elle. Cela aide à créer des technologies plus conviviales et éthiques. Une communication courtoise améliore également l'interaction avec le réseau de neurones.
    `

    },
    ar: {
      title: "المفاهيم الأساسية للشبكات العصبية",
    description: `
      **ما هي الشبكات العصبية؟** الشبكة العصبية هي برنامج حاسوبي يحاكي عمل الدماغ البشري. تتكون من العديد من العناصر الصغيرة التي تسمى العصبونات، والتي تعمل معًا لمعالجة المعلومات وحل المشكلات. على سبيل المثال، يمكن للشبكة العصبية التعرف على الصور، وفهم النصوص، أو التنبؤ بالنتائج بناءً على البيانات.
      
      **تاريخ الشبكات العصبية:**
      1. عام 1943: وارن مكولوتش ووالتر بيتس قاما بإنشاء أول نموذج نظري للعصبون الاصطناعي بناءً على العصبونات البيولوجية.
      2. عام 1958: فرانك روزينبلات طور البرسيبترون، وهو شبكة عصبية بسيطة يمكنها التعلم والتنبؤ بناءً على بيانات الإدخال.
      3. الستينيات: تطوير البرسيبترونات متعددة الطبقات (MLP)، مما سمح بإنشاء نماذج أكثر تعقيدًا وقوة للشبكات العصبية.
      4. الثمانينيات: جيفري هينتون وزملاؤه طوروا طريقة انتشار الأخطاء العكسي، مما جعل تدريب الشبكات العصبية متعددة الطبقات أكثر فعالية.
      5. التسعينيات: يان ليكون أدخل الشبكات العصبية الالتفافية (CNN) للتعرف على الأرقام المكتوبة باليد، مما أسس لأنظمة التعرف على الصور الحديثة.
      6. عام 2006: جيفري هينتون قدم مفهوم التعلم العميق، مما أدى إلى تحسين كبير في أداء الشبكات العصبية.
      7. عام 2012: تقدم كبير في مجال التعرف على الصور بفضل شبكة AlexNet، التي فازت بمسابقة ImageNet وأظهرت نتائج فائقة في رؤية الحاسوب.
      8. منذ عام 2014: إيان غودفيلو طور الشبكات التوليدية التنافسية (GAN)، مما فتح آفاقًا جديدة لتوليد الصور والنصوص الواقعية.
      9. العصر الحديث: التحسين المستمر لهياكل الشبكات العصبية، مثل المحولات وBERT، التي تسمح بتحقيق إنجازات جديدة في مختلف تطبيقات الذكاء الاصطناعي.
      
      **المفاهيم الأساسية:**
      1. العصبونات والطبقات: تتكون الشبكة العصبية من عصبونات مرتبة في طبقات. هناك ثلاثة أنواع من الطبقات: طبقة الإدخال، الطبقات المخفية، وطبقة الإخراج. تستقبل طبقة الإدخال البيانات، تعالج الطبقات المخفية المعلومات، وتقدم طبقة الإخراج النتيجة.
      2. الأوزان والوصلات: كل عصبون مرتبط بالعصبونات الأخرى من خلال الأوزان. تحدد الأوزان أهمية كل إشارة وتضبط أثناء التدريب لكي تتمكن الشبكة العصبية من التنبؤ بدقة.
      3. وظائف التنشيط: تساعد هذه الوظائف العصبونات في اتخاذ القرارات عن طريق تحويل الإشارات المدخلة إلى مخرجات. أمثلة على وظائف التنشيط تشمل الدالة السينية، ReLU (الوحدة الخطية المصححة) وtanh (الظل الزائدي).
      4. التدريب وانتشار الأخطاء العكسي: تتدرب الشبكة العصبية بناءً على بيانات الإدخال. خلال التدريب، تضبط أوزانها باستخدام طريقة انتشار الأخطاء العكسي لتقليل الفجوة بين القيم المتنبأة والفعلية (الخطأ).
      
      **عملية عمل الشبكات العصبية:**
      1. إدخال البيانات: يتم إدخال بيانات الإدخال (مثل الصور، النصوص) في طبقة الإدخال للشبكة العصبية.
      2. نقل الإشارات: تمر الإشارات عبر الأوزان ووظائف التنشيط في الطبقات المخفية، حيث تتم معالجة المعلومات وتحويلها.
      3. طبقة الإخراج: يتم نقل الإشارات المعالجة إلى طبقة الإخراج، التي تقدم النتيجة (مثل التعرف على الصور أو التنبؤ بالنصوص).
      4. التدريب: تتدرب الشبكة العصبية عن طريق ضبط أوزانها باستخدام طريقة انتشار الأخطاء العكسي لتقليل الخطأ بين القيم المتنبأة والفعلية.
      5. التقييم: بعد التدريب، يتم اختبار الشبكة العصبية على بيانات جديدة لتقييم أدائها ودقتها.
      
      **أمثلة على الشبكات العصبية المجانية:**
      1. OpenAI GPT-3: نموذج قوي لتوليد النصوص، يمكنه المساعدة في كتابة المقالات، صياغة الرسائل أو الإجابة على الأسئلة.
      2. Google Colab: منصة مجانية تدعم دفاتر Jupyter مع دعم GPU، مما يتيح تطوير وتدريب نماذج التعلم الآلي مجانًا.
      3. Hugging Face: منصة للعمل مع البيانات النصية، تتيح استخدام النماذج المدربة مسبقًا لترجمة النصوص، تحليل المشاعر والمهام الأخرى.
      
      **أمثلة على الشبكات العصبية المدفوعة:**
      1. Microsoft Azure AI: خدمات سحابية تتيح إنشاء ونشر الشبكات العصبية لمهام متنوعة، بما في ذلك تحليل الصور والنصوص.
      2. AWS AI: مجموعة من الأدوات من أمازون للتعلم الآلي والشبكات العصبية، تشمل خدمات لمعالجة الصوت، النصوص والصور.
      3. IBM Watson: أدوات متقدمة لتحليل البيانات وتطوير التطبيقات الذكية، تستخدم على نطاق واسع في الطب، التمويل وغيرها.
      
      **ما الذي يمكن أن تفعله الشبكات العصبية؟**
      1. التعرف على الصور: يمكن للشبكة العصبية التعرف على الأشياء في الصور. على سبيل المثال، يستخدم Facebook الشبكات العصبية للتعرف على الأصدقاء في الصور المحملة ووضع العلامات عليهم تلقائيًا.
      2. معالجة اللغة الطبيعية (NLP): يمكن للشبكة العصبية ترجمة النصوص من لغة إلى أخرى. يستخدم Google Translate الشبكات العصبية لتحسين دقة وطبيعة الترجمات.
      3. أنظمة التوصية: يمكن للشبكة العصبية تقديم توصيات للأفلام بناءً على تفضيلاتك. يستخدم Netflix الشبكات العصبية لتحليل سجلات المشاهدة للأفلام والمسلسلات وتقديم المحتوى الذي قد يعجبك.
      4. تشخيص الأمراض: يمكن للشبكة العصبية تحليل الصور الطبية للكشف عن الأمراض. تساعد الشبكة العصبية التي طورتها Google Health الأطباء في اكتشاف سرطان الرئة في مراحله المبكرة بدقة عالية.
      5. أتمتة التداول: يمكن للشبكة العصبية التنبؤ باتجاهات السوق وتنفيذ العمليات التجارية. تستخدم الشركات المالية الشبكات العصبية لتحليل كميات كبيرة من بيانات السوق وتنفيذ الصفقات تلقائيًا.
      6. تحليل المشاعر: يمكن للشبكة العصبية تحديد النغمة العاطفية للنصوص. على سبيل المثال، يستخدم Twitter الشبكات العصبية لتحليل مشاعر التغريدات واكتشاف ردود الفعل العامة على الأحداث.
      7. إنشاء صور صناعية: يمكن للشبكة العصبية توليد صور واقعية. يمكن لـ GAN التي طورتها NVIDIA إنشاء صور فوتوغرافية واقعية لأشخاص غير موجودين بالفعل.
      
      **آداب التواصل مع الذكاء الاصطناعي:** على الرغم من أن الذكاء الاصطناعي لا يمتلك مشاعر، إلا أن التحدث معه بلطف مهم. يساعد ذلك في إنشاء تقنيات أكثر ودية وأخلاقية. كما أن التواصل اللطيف يحسن التفاعل مع الشبكة العصبية.
    `

    },
    bn: {
        title: "নিউরাল নেটওয়ার্কের মৌলিক ধারণা",
    description: `
      **নিউরাল নেটওয়ার্ক কি?** নিউরাল নেটওয়ার্ক হল একটি কম্পিউটার প্রোগ্রাম যা মানব মস্তিষ্কের কার্যকারিতা অনুকরণ করে। এটি অনেক ছোট উপাদান দ্বারা গঠিত, যেগুলি একত্রে তথ্য প্রক্রিয়া করে এবং সমস্যার সমাধান করে। উদাহরণস্বরূপ, একটি নিউরাল নেটওয়ার্ক ছবি চিনতে পারে, পাঠ্য বুঝতে পারে বা ডেটার উপর ভিত্তি করে ফলাফল পূর্বানুমান করতে পারে।
      
      **নিউরাল নেটওয়ার্কের ইতিহাস:**
      1. 1943: ওয়ারেন ম্যাককালক এবং ওয়াল্টার পিটস প্রথম সেরা তথ্যমূলক মডেল তৈরি করেন যা জৈবিক নিউরনের উপর ভিত্তি করে।
      2. 1958: ফ্রাঙ্ক রোজেনব্লাট একটি সাধারণ নিউরাল নেটওয়ার্ক ডেভেলপ করেন, যেটি ইনপুট ডেটার উপর ভিত্তি করে শিখতে এবং পূর্বানুমান করতে পারে।
      3. 1960-এর দশক: বহুপ্লক নিউরাল নেটওয়ার্ক (MLP) বিকাশ হয়, যা আরও জটিল এবং শক্তিশালী মডেল তৈরির জন্য সুযোগ তৈরি করে।
      4. 1980-এর দশক: জেফ্রি হিন্টন এবং তার সহযোগীরা ভুল ব্যাকপ্রোপাগেশন পদ্ধতি ডেভেলপ করেন, যা বহুপ্লক নিউরাল নেটওয়ার্কের প্রশিক্ষণকে আরও কার্যকর করে তোলে।
      5. 1990-এর দশক: ইয়ান লেকান হ্যান্ডরিটেন সংখ্যা স্বীকৃতির জন্য কনভলিউশনাল নিউরাল নেটওয়ার্ক (CNN) পরিচয় করান, যা আধুনিক চিত্র স্বীকৃতি সিস্টেমের ভিত্তি সৃষ্টি করে।
      6. 2006: জেফ্রি হিন্টন গভীর শিক্ষার ধারণা তুলে ধরেন, যা নিউরাল নেটওয়ার্কের কর্মক্ষমতাকে উল্লেখযোগ্যভাবে উন্নত করে।
      7. 2012: ইমেজনেট প্রতিযোগিতায় AlexNet এর সাফল্যের কারণে চিত্র স্বীকৃতির ক্ষেত্রে একটি বিপ্লব ঘটে।
      8. 2014 থেকে: ইয়ান গুডফেলো দ্বারা জেনারেটিভ অ্যাডভার্সিয়াল নেটওয়ার্ক (GAN) এর বিকাশ ঘটে, যা বাস্তবসম্মত চিত্র এবং পাঠ্য তৈরি করতে নতুন সম্ভাবনা তৈরি করে।
      9. আধুনিক যুগ: ট্রান্সফরমার এবং BERT এর মতো নিউরাল নেটওয়ার্ক আর্কিটেকচারের ক্রমাগত উন্নতি, যা বিভিন্ন এআই অ্যাপ্লিকেশনের ক্ষেত্রে নতুন উচ্চতা অর্জনের সুযোগ দেয়।
      
      **মৌলিক ধারণা:**
      1. নিউরন এবং স্তর: একটি নিউরাল নেটওয়ার্ক নিউরন দ্বারা গঠিত, যেগুলি স্তরে সংগঠিত থাকে। তিনটি স্তরের প্রকার রয়েছে: ইনপুট স্তর, লুকানো স্তর এবং আউটপুট স্তর। ইনপুট স্তর তথ্য গ্রহণ করে, লুকানো স্তরগুলি তথ্য প্রক্রিয়া করে এবং আউটপুট স্তর ফলাফল প্রদান করে।
      2. ওজন এবং সংযোগ: প্রতিটি নিউরন ওজনের মাধ্যমে অন্যান্য নিউরনের সাথে সংযুক্ত থাকে। ওজন প্রতিটি সংকেতের গুরুত্ব নির্ধারণ করে এবং প্রশিক্ষণের সময় সমন্বিত হয় যাতে নিউরাল নেটওয়ার্ক সঠিক পূর্বানুমান করতে পারে।
      3. সক্রিয়করণ ফাংশন: এই ফাংশনগুলি ইনপুট সংকেতগুলি আউটপুটে রূপান্তরিত করে নিউরনগুলিকে সিদ্ধান্ত নিতে সহায়তা করে। সক্রিয়করণ ফাংশনের উদাহরণগুলির মধ্যে রয়েছে সিগময়েড ফাংশন, ReLU (আলাদা রেখাচিত্রমালা একক) এবং tanh (হাইপারবোলিক ট্যাঞ্জেন্ট)।
      4. প্রশিক্ষণ এবং ভুল ব্যাকপ্রোপাগেশন: নিউরাল নেটওয়ার্ক ইনপুট ডেটার উপর ভিত্তি করে প্রশিক্ষিত হয়। প্রশিক্ষণের সময়, এটি এর ওজনগুলি সমন্বিত করে ভুল ব্যাকপ্রোপাগেশন পদ্ধতির মাধ্যমে পূর্বানুমান এবং প্রকৃত মানের মধ্যে পার্থক্য (ভুল) কমাতে।
      
      **নিউরাল নেটওয়ার্কের কাজের প্রক্রিয়া:**
      1. ডেটা ইনপুট: ইনপুট ডেটা (যেমন, ছবি, পাঠ্য) নিউরাল নেটওয়ার্কের ইনপুট স্তরে প্রবেশ করা হয়।
      2. সংকেত প্রেরণ: সংকেত ওজন এবং সক্রিয়করণ ফাংশনের মাধ্যমে লুকানো স্তরে প্রেরিত হয়, যেখানে তথ্য প্রক্রিয়া এবং রূপান্তরিত হয়।
      3. আউটপুট স্তর: প্রক্রিয়াকৃত সংকেত আউটপুট স্তরে প্রেরিত হয়, যা ফলাফল প্রদান করে (যেমন, ছবি স্বীকৃতি বা পাঠ্য পূর্বানুমান)।
      4. প্রশিক্ষণ: ভুল ব্যাকপ্রোপাগেশন পদ্ধতির মাধ্যমে ওজনগুলি সমন্বিত করে প্রশিক্ষণ দেওয়া হয় যাতে পূর্বানুমান এবং প্রকৃত মানের মধ্যে ভুল কমানো যায়।
      5. মূল্যায়ন: প্রশিক্ষণের পরে, নিউরাল নেটওয়ার্ক নতুন ডেটার উপর পরীক্ষা করা হয় এর কর্মক্ষমতা এবং যথার্থতা মূল্যায়ন করতে।
      
      **মুক্ত নিউরাল নেটওয়ার্কের উদাহরণ:**
      1. OpenAI GPT-3: একটি শক্তিশালী পাঠ্য জেনারেটর মডেল, যা প্রবন্ধ, পত্র বা প্রশ্নের উত্তর লিখতে সাহায্য করতে পারে।
      2. Google Colab: একটি বিনামূল্যে জুপাইটার নোটবুক প্ল্যাটফর্ম যা GPU সমর্থন সহ, যা বিনামূল্যে মেশিন লার্নিং মডেলগুলি উন্নত এবং প্রশিক্ষণ দেওয়ার সুযোগ দেয়।
      3. Hugging Face: একটি পাঠ্য ডেটা প্রক্রিয়াকরণ প্ল্যাটফর্ম, যেখানে আপনি পূর্বানুমিত মডেলগুলি ব্যবহার করতে পারেন পাঠ্য অনুবাদ, অনুভূতির বিশ্লেষণ এবং অন্যান্য কাজের জন্য।
      
      **প্রিমিয়াম নিউরাল নেটওয়ার্কের উদাহরণ:**
      1. Microsoft Azure AI: ক্লাউড পরিষেবা, যা বিভিন্ন কাজের জন্য নিউরাল নেটওয়ার্কগুলি তৈরি এবং স্থাপন করতে দেয়, যার মধ্যে রয়েছে চিত্র এবং পাঠ্য বিশ্লেষণ।
      2. AWS AI: Amazon এর একটি সেট টুলস, যা মেশিন লার্নিং এবং নিউরাল নেটওয়ার্কের জন্য, ভাষণ, পাঠ্য এবং চিত্র প্রক্রিয়াকরণ পরিষেবার সাথে।
      3. IBM Watson: উন্নত ডেটা বিশ্লেষণ এবং স্মার্ট অ্যাপ্লিকেশন ডেভেলপমেন্ট টুলস, যা ব্যাপকভাবে ব্যবহৃত হয় চিকিৎসা, আর্থিক এবং অন্যান্য ক্ষেত্রগুলিতে।
      
      **নিউরাল নেটওয়ার্ক কি করতে পারে?**
      1. চিত্র স্বীকৃতি: নিউরাল নেটওয়ার্ক ছবি চিহ্নিত করতে পারে। উদাহরণস্বরূপ, Facebook আপলোড করা ছবিতে বন্ধুদের স্বয়ংক্রিয় স্বীকৃতি এবং ট্যাগিং করতে নিউরাল নেটওয়ার্ক ব্যবহার করে।
      2. প্রাকৃতিক ভাষা প্রক্রিয়াকরণ (NLP): নিউরাল নেটওয়ার্ক পাঠ্য একটি ভাষা থেকে অন্য ভাষায় অনুবাদ করতে পারে। Google Translate নিউরাল নেটওয়ার্ক ব্যবহার করে অনুবাদের যথার্থতা এবং স্বাভাবিকতা বৃদ্ধি করে।
      3. সুপারিশ সিস্টেম: নিউরাল নেটওয়ার্ক আপনার পছন্দের উপর ভিত্তি করে সিনেমা সুপারিশ করতে পারে। Netflix নিউরাল নেটওয়ার্ক ব্যবহার করে সিনেমা এবং টিভি শো দেখার রেকর্ড বিশ্লেষণ করে এবং আপনার পছন্দের বিষয়বস্তু সুপারিশ করে।
      4. রোগ নির্ণয়: নিউরাল নেটওয়ার্ক চিকিৎসা ছবি বিশ্লেষণ করতে পারে রোগ নির্ণয় করতে। Google Health দ্বারা বিকাশিত নিউরাল নেটওয়ার্ক উচ্চ নির্ভুলতার সাথে প্রাথমিক পর্যায়ে ফুসফুসের ক্যান্সার সনাক্ত করতে ডাক্তারদের সহায়তা করে।
      5. ট্রেড অটোমেশন: নিউরাল নেটওয়ার্ক বাজারের প্রবণতা পূর্বানুমান করতে পারে এবং বাণিজ্য অপারেশন সম্পাদন করতে পারে। আর্থিক কোম্পানিগুলি বড় আকারের বাজার ডেটা বিশ্লেষণ করতে এবং
    },
    pt: {
        title: "Conceitos básicos de redes neurais",
    description: `
      **O que são redes neurais?** Uma rede neural é um programa de computador que imita o funcionamento do cérebro humano. Ela consiste em muitos pequenos elementos chamados neurônios, que trabalham juntos para processar informações e resolver problemas. Por exemplo, uma rede neural pode reconhecer imagens, entender texto ou prever resultados com base em dados.
      
      **História das redes neurais:**
      1. 1943: Warren McCulloch e Walter Pitts criaram o primeiro modelo teórico de um neurônio artificial, baseado em neurônios biológicos.
      2. 1958: Frank Rosenblatt desenvolveu o perceptron, uma rede neural simples que podia aprender e fazer previsões com base em dados de entrada.
      3. Década de 1960: Desenvolvimento dos perceptrons multicamadas (MLP), o que permitiu criar modelos mais complexos e poderosos de redes neurais.
      4. Década de 1980: Geoffrey Hinton e seus colegas desenvolveram o método de retropropagação de erros, o que permitiu treinar redes neurais multicamadas de forma mais eficaz.
      5. Década de 1990: Yann LeCun introduziu as redes neurais convolucionais (CNN) para reconhecimento de dígitos manuscritos, estabelecendo as bases para os sistemas modernos de reconhecimento de imagens.
      6. 2006: Geoffrey Hinton apresentou o conceito de aprendizado profundo, o que levou a uma melhoria significativa no desempenho das redes neurais.
      7. 2012: Avanço no reconhecimento de imagens graças à rede AlexNet, que venceu o concurso ImageNet e demonstrou resultados superiores na área de visão computacional.
      8. Desde 2014: Desenvolvimento de redes adversárias geradoras (GAN) por Ian Goodfellow, o que abriu novas possibilidades para a geração de imagens e texto realistas.
      9. Atualidade: Melhoria contínua das arquiteturas de redes neurais, como transformers e BERT, permitindo alcançar novos patamares em diversas aplicações de IA.
      
      **Conceitos básicos:**
      1. Neurônios e camadas: Uma rede neural é composta de neurônios organizados em camadas. Existem três tipos de camadas: entrada, ocultas e saída. A camada de entrada recebe os dados, as camadas ocultas processam a informação e a camada de saída fornece o resultado.
      2. Pesos e conexões: Cada neurônio está conectado a outros neurônios através de pesos. Os pesos determinam a importância de cada sinal e são ajustados durante o treinamento para que a rede neural possa fazer previsões precisas.
      3. Funções de ativação: Essas funções ajudam os neurônios a tomar decisões, transformando os sinais de entrada em saída. Exemplos de funções de ativação incluem a função sigmóide, ReLU (Unidade Linear Retificada) e tanh (tangente hiperbólica).
      4. Treinamento e retropropagação de erros: A rede neural é treinada com base em dados de entrada. Durante o treinamento, ajusta seus pesos usando o método de retropropagação de erros para minimizar a diferença entre os valores previstos e reais (erro).
      
      **Processo de funcionamento das redes neurais:**
      1. Entrada de dados: Os dados de entrada (por exemplo, imagens, texto) são inseridos na camada de entrada da rede neural.
      2. Transmissão de sinais: Os sinais passam pelos pesos e pelas funções de ativação nas camadas ocultas, onde a informação é processada e transformada.
      3. Camada de saída: Os sinais processados são transmitidos à camada de saída, que fornece o resultado (por exemplo, reconhecimento de imagens ou previsão de texto).
      4. Treinamento: A rede neural é treinada ajustando seus pesos usando o método de retropropagação de erros para reduzir o erro entre os valores previstos e reais.
      5. Avaliação: Após o treinamento, a rede neural é testada com novos dados para avaliar seu desempenho e precisão.
      
      **Exemplos de redes neurais gratuitas:**
      1. OpenAI GPT-3: Um modelo poderoso de geração de texto que pode ajudar a escrever artigos, redigir cartas ou responder perguntas.
      2. Google Colab: Uma plataforma gratuita que suporta notebooks Jupyter com suporte a GPU, permitindo desenvolver e treinar modelos de aprendizado de máquina gratuitamente.
      3. Hugging Face: Uma plataforma para trabalhar com dados de texto, permitindo usar modelos pré-treinados para tradução de texto, análise de sentimentos e outras tarefas.
      
      **Exemplos de redes neurais pagas:**
      1. Microsoft Azure AI: Serviços em nuvem que permitem criar e implementar redes neurais para diversas tarefas, incluindo análise de imagens e texto.
      2. AWS AI: Um conjunto de ferramentas da Amazon para aprendizado de máquina e redes neurais, incluindo serviços para processamento de fala, texto e imagens.
      3. IBM Watson: Ferramentas avançadas para análise de dados e desenvolvimento de aplicações inteligentes, amplamente utilizadas em medicina, finanças e outros campos.
      
      **O que as redes neurais podem fazer?**
      1. Reconhecimento de imagens: Uma rede neural pode reconhecer objetos em fotografias. Por exemplo, o Facebook usa redes neurais para reconhecer e marcar automaticamente amigos nas fotos carregadas.
      2. Processamento de linguagem natural (NLP): Uma rede neural pode traduzir texto de um idioma para outro. O Google Translate usa redes neurais para melhorar a precisão e a naturalidade das traduções.
      3. Sistemas de recomendação: Uma rede neural pode recomendar filmes com base em suas preferências. A Netflix usa redes neurais para analisar registros de visualização de filmes e séries e recomendar conteúdo que você possa gostar.
      4. Diagnóstico de doenças: Uma rede neural pode analisar imagens médicas para detectar doenças. Uma rede neural desenvolvida pelo Google Health ajuda médicos a detectar câncer de pulmão em estágios iniciais com alta precisão.
      5. Automação de comércio: Uma rede neural pode prever tendências de mercado e executar operações comerciais. Empresas financeiras usam redes neurais para analisar grandes volumes de dados de mercado e realizar operações automaticamente.
      6. Análise de sentimentos: Uma rede neural pode determinar o tom emocional de um texto. Por exemplo, o Twitter usa redes neurais para analisar o sentimento dos tweets e detectar as reações do público a eventos.
      7. Criação de imagens artificiais: Uma rede neural pode gerar imagens realistas. A GAN desenvolvida pela NVIDIA pode criar imagens fotorrealistas de pessoas que não existem na realidade.
      
      **Cortesia na comunicação com a IA:** Embora a IA não tenha sentimentos, é importante ser cortês ao se comunicar com ela. Isso ajuda a criar tecnologias mais amigáveis e éticas. A comunicação cortês também melhora a interação com a rede neural.
    `

    },
    ru: {
        title: "Основные концепции работы нейросетей",
        description: `
        **Что такое нейросети?**
        Нейросеть — это компьютерная программа, которая имитирует работу человеческого мозга. Она состоит из множества небольших элементов, называемых нейронами, которые работают вместе, чтобы обрабатывать информацию и решать задачи. Например, нейросеть может распознавать изображения, понимать текст или предсказывать результаты на основе данных.

        **История нейросетей:**
        1. 1943 год: Уоррен МакКаллок и Уолтер Питтс создали первую теоретическую модель искусственного нейрона, основываясь на биологических нейронах.
        2. 1958 год: Фрэнк Розенблатт разработал персептрон — простую нейросеть, которая могла обучаться и делать прогнозы на основе входных данных.
        3. 1960-е годы: Разработка многослойных персептронов (MLP), что позволило создать более сложные и мощные модели нейросетей.
        4. 1980-е годы: Джеффри Хинтон и его коллеги разработали метод обратного распространения ошибок, что позволило более эффективно обучать многослойные нейросети.
        5. 1990-е годы: Введение свёрточных нейросетей (CNN) Яном Лекуном для распознавания рукописных цифр, что стало основой для современных систем распознавания изображений.
        6. 2006 год: Джеффри Хинтон представил концепцию глубокого обучения, что привело к значительному улучшению производительности нейросетей.
        7. 2012 год: Прорыв в области распознавания изображений благодаря сети AlexNet, которая выиграла конкурс ImageNet и продемонстрировала превосходные результаты в области компьютерного зрения.
        8. С 2014 года: Разработка генеративно-состязательных сетей (GAN) Ианом Гудфеллоу, что открыло новые возможности для генерации реалистичных изображений и текста.
        9. Современность: Продолжающееся улучшение архитектур нейросетей, таких как трансформеры и BERT, что позволяет достигать новых высот в различных областях применения ИИ.

        **Основные концепции:**
        1. Нейроны и слои: Нейросеть состоит из нейронов, организованных в слои. Существует три типа слоев: входной, скрытые и выходной. Входной слой принимает данные, скрытые слои обрабатывают информацию, а выходной слой выдает результат.
        2. Веса и связи: Каждый нейрон связан с другими нейронами через веса. Веса определяют важность каждого сигнала и корректируются во время обучения, чтобы нейросеть могла делать точные предсказания.
        3. Активационные функции: Эти функции помогают нейронам принимать решения, преобразуя входные сигналы в выходные. Примеры активационных функций включают сигмоидную функцию, ReLU (Rectified Linear Unit) и tanh (гиперболический тангенс).
        4. Обучение и обратное распространение ошибок: Нейросеть обучается на основе входных данных. В процессе обучения она корректирует свои веса с помощью метода обратного распространения ошибок, чтобы минимизировать разницу между предсказанными и реальными значениями (ошибку).

        **Процесс работы нейросетей:**
        1. Ввод данных: Входные данные (например, изображения, текст) подаются на входной слой нейросети.
        2. Передача сигналов: Сигналы проходят через веса и активационные функции в скрытых слоях, где информация обрабатывается и преобразуется.
        3. Выходной слой: Обработанные сигналы передаются на выходной слой, который выдает результат (например, распознавание изображения или предсказание текста).
        4. Обучение: Нейросеть обучается, корректируя свои веса с помощью метода обратного распространения ошибок, чтобы уменьшить ошибку между предсказанными и реальными значениями.
        5. Оценка: После обучения нейросеть проверяется на новых данных, чтобы оценить её производительность и точность.

        **Примеры бесплатных нейросетей:**
        1. OpenAI GPT-3: Мощная модель для генерации текста. Она может помочь написать статьи, составить письма или ответить на вопросы.
        2. Google Colab: Бесплатная платформа для выполнения Jupyter notebooks с поддержкой GPU, что позволяет разрабатывать и обучать модели машинного обучения бесплатно.
        3. Hugging Face: Платформа для работы с текстовыми данными. Здесь можно использовать готовые модели для перевода текста, анализа тональности и других задач.

        **Примеры платных нейросетей:**
        1. Microsoft Azure AI: Облачные сервисы, которые позволяют создавать и развертывать нейросети для различных задач, включая анализ изображений и текста.
        2. AWS AI: Набор инструментов от Amazon для машинного обучения и нейросетей. Сюда входят сервисы для обработки речи, текста и изображений.
        3. IBM Watson: Продвинутые инструменты для анализа данных и создания умных приложений. Watson используется в медицине, финансах и других областях.

        **Что могут нейросети?**
        1. Распознавание изображений: Нейросеть может распознавать объекты на фотографиях. Например, в Facebook используется нейросеть для автоматического распознавания и отметки друзей на загруженных фотографиях.
        2. Обработка естественного языка (NLP): Нейросеть может переводить текст с одного языка на другой. Google Translate использует нейросети для повышения точности и естественности переводов.
        3. Рекомендательные системы: Нейросеть может рекомендовать фильмы на основе ваших предпочтений. Netflix использует нейросети для анализа просмотра фильмов и сериалов, чтобы предложить вам контент, который вам может понравиться.
        4. Диагностика заболеваний: Нейросеть может анализировать медицинские снимки для выявления заболеваний. Нейросеть, разработанная Google Health, помогает врачам обнаруживать рак легких на ранних стадиях с высокой точностью.
        5. Автоматизация торговли: Нейросеть может прогнозировать рыночные тренды и выполнять торговые операции. В финансовых компаниях используются нейросети для анализа больших объемов рыночных данных и автоматического проведения сделок.
        6. Анализ настроений: Нейросеть может определять эмоциональную окраску текста. Например, в Twitter используется нейросеть для анализа тональности твитов, чтобы выявить общественные настроения и реакции на события.
        7. Создание искусственных изображений: Нейросеть может генерировать реалистичные изображения. GAN, разработанная компанией NVIDIA, может создавать фотореалистичные изображения людей, которые на самом деле не существуют.

        **Вежливость в общении с ИИ:**
        Хотя ИИ не обладает чувствами, вежливость при общении с ним важна. Это помогает создавать более дружелюбные и этичные технологии. Вежливое общение также улучшает взаимодействие с нейросетью.
    };
     document.querySelectorAll('.language-btn').forEach(button => {
    button.addEventListener('click', (event) => {
        const selectedLang = event.target.getAttribute('data-lang');
        const translation = translations[selectedLang];
        if (translation) {
            document.getElementById('title').textContent = translation.title;
            document.getElementById('description').textContent = translation.description;
        }
    });
});

document.getElementById('light-mode').addEventListener('click', () => {
    document.body.classList.remove('dark-mode');
    document.body.classList.add('light-mode');
});

document.getElementById('dark-mode').addEventListener('click', () => {
    document.body.classList.remove('light-mode');
    document.body.classList.add('dark-mode');
});
