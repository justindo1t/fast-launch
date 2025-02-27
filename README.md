# fast-launch

# Basic/Small Tool
## Feature Enrichment
1. Time Management
   - arrow
3. Path Management
   - pathlib
4. Log Management
   - loguru / logging
5. processbar Management (Feature)
   - tqdm / panda-parallel
6. argparse / typer / click (命令行工具构建库) (Feature)
7. python-dotenv (从 `.env` 文件中加载环境变量)

## File IO (pyyaml, json, pandas) (pdf) (openpyxl / xlrd / xlwt) (h5py)

1. PyYAML / json

## Parallel

1. multiprocessing / asyncio

   

## Sci-compute
| Numpy, Pandas, Scipy,
Vis (plotly)
| matplotlib, Seaborn, Bokeh, ... (R)

## ML / DL
1. scikit-learn
2. XGBoost / LightGBM / CatBoost
3. auto-sklearn (自动化调参)
4. auto-pands (Auto-EDA)

1. PyTorch | einops
2. Huggingface | transformer | tokenizer | datasets
3. Jax
4. TensorFlow | Keras (For run other's SOTA model)

1. wandb, tensorboard

| (hydra) 配置管理框架，在多环境、多配置实验中非常便利

1. pytorch-lightning

## traditional NLP
  a. spaCy (tokenize, tag词性标注, NER, parse依存分析)
  b. NLTK (tokenize, stemming, grammar analytic)
  c. Gensim (LDA, Word2vec, FastText)
  d. SentencePiece (BPE, Unigram)

## Web-dev
1. Flask / Django (Web Framework) 重量级 Web 框架，内置管理后台、ORM、会话管理等 (场景：传统网站、完整后端服务；适合大型项目)
   - 轻量级 Web 框架，易于上手，插件丰富 (场景：快速搭建小型后端服务、API、原型应用)
2. FastAPI (?) (功能：新兴高性能 Web 框架，基于 Python 的异步特性，自动生成 API 文档) ()

## 爬虫 / Web Request / Data Collections
| Request, aiohttp (?) (asyncio-based 异步 HTTP Client/Server) (需要大规模并发或异步请求时)

1. Beautiful Soup (bs4)
2. Scrapy (爬虫框架)
3. Selenium

## Database / Data Storage

1. mysql

   	1.	SQLAlchemy
	•	功能：Python ORM，支持与多种数据库（MySQL、PostgreSQL、SQLite 等）交互。
	•	场景：需要更抽象的数据库操作层；项目规模较大或可维护性要求高。
	2.	Peewee
	•	功能：相对轻量级的 ORM，学习曲线更平缓。
	•	场景：中小型项目，或需要简单 ORM 功能。
	3.	PyMySQL / psycopg2 / cx_Oracle
	•	功能：针对 MySQL、PostgreSQL、Oracle 等数据库的原生连接驱动库。
	•	场景：需要进行更底层的 SQL 操作；对特性/性能有更精细掌控。
	4.	HDF5 (h5py)
	•	功能：对 HDF5 格式文件进行读写，处理多维数组存储。
	•	场景：大规模科学计算数据存储，如深度学习训练数据。
	5.	pyarrow.parquet
	•	功能：读写 Parquet 列式存储格式的 Python 接口。
	•	场景：与大数据生态（Spark、Hive、Trino 等）交互，共享高效存储格式。

## Unit Test / Quality Control
1. pytest / unittest (Unit Test) (Integrration Test) (Automation Test)

coverage
	•	功能：测试覆盖率统计工具，结合 pytest/unittest 使用。
	•	场景：评估测试完整度，查找未被测试代码行

| pylint, flake8, black: 代码格式化与静态分析工具；black用于自动格式化，pylint/flake8 提示风格问题
| 场景：团队协作时代码风格规范，提高可读性

## Container / Cloud ecosystem (optional)
	1.	Docker SDK for Python
	•	功能：用 Python 脚本管理 Docker 容器、镜像。
	•	场景：自动化部署、CI/CD。
	2.	boto3
	•	功能：AWS 服务的官方 Python SDK。
	•	场景：在 AWS 上使用 S3、EC2、Lambda 等服务时。
	3.	google-cloud-python
	•	功能：GCP 各种服务(Compute, Storage, BigQuery等)的 SDK。
	•	场景：在 Google Cloud 平台上部署与管理资源。

# CV
1. opencv|cv2, pillow

# Dashboard / Web-app
Streamlit / Gradio / Dash：快速构建可交互的 Web 应用或仪表盘，尤其适合数据科学原型演示

# RL 
gym / ray[rllib]：强化学习环境与分布式训练框架。
