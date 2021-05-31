import pandas
import numpy
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

payload_data = pandas.read_csv("C:/Users/root/Downloads/archive/sqli.csv",
                               error_bad_lines=False,
                               sep=',',
                               encoding='utf-16')

print(payload_data.info())
payload_data['Sentence'] = payload_data['Sentence'].astype(str)
payload_train, payload_test, y_train, y_test = train_test_split(payload_data['Sentence'],
    payload_data['Label'], test_size=0.25, shuffle=False, random_state=23)

stopwords = ['a', 'an']

X_train = []
for stc in payload_train:
    token = []
    words = stc.split()
    for word in words:
        token.append(word.lower())
    X_train.append(token)

X_test = []
for stc in payload_test:
    token = []
    words = stc.split()
    for word in words:
        if word not in stopwords:
            token.append(word.lower())
    X_test.append(token)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(len(tokenizer.word_counts)) # 8543

count = 0
for word, word_count in tokenizer.word_counts.items():
    if word_count > 1:
        count += 1
print(count) # 3887

tokenizer = Tokenizer(4000)
tokenizer.fit_on_texts(X_train)

# 부여된 정수 인덱스로 변환
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# 데이터 패딩
max_len = max(len(l) for l in X_train)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# 모델 구축
model = Sequential()
model.add(Embedding(4000, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
model_check = ModelCheckpoint('the_best.h5', monitor='val_acc', mode='max', verbose=1,
                              save_best_only=True)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64,
          callbacks=[early_stop, model_check])

print(model.evaluate(X_test, y_test))

while True:
    print('ID : ', end=' ')
    user_input = input().split()
    user_data = [[]]
    for word in user_input:
        user_data[0].append(word.lower())
    user_data = tokenizer.texts_to_sequences(user_data)
    user_data = pad_sequences(user_data, maxlen=max_len)

    if (model.predict(user_data) > 0.5):
        print(f"{user_input}")
        print(" 해킹 시도입니다.")
    else:
        print(f"{user_input}")
        print(" 해킹 시도가 아닙니다.")