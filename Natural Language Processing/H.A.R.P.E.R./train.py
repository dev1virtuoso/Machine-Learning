from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 加载预训练的GPT-2模型和分词器
model_name = 'gpt2'  # 或者其他合适的预训练模型
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# 准备训练数据
train_file = '/Users/tszsanwu/Developer/Code/H.A.R.P.E.R./Data/user_style_dataset.txt'  # 用户独特风格的文本数据集
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
)

# 创建Trainer对象并进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)
trainer.train()
