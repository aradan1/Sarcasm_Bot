
# coding: utf-8
import classifier
import datasetExtractor


import asyncio as aio
import telepot

from telepot.aio.loop import MessageLoop
from telepot.aio.delegate import pave_event_space, per_chat_id, create_open, include_callback_query_chat_id



class SarcasmBot(object):
    
    instance = None
    
    def __init__(self):
        assert SarcasmBot.instance is None
        SarcasmBot.instance = self
        
        self.bot = None
        self.loop = aio.get_event_loop()
        self.msg_loop = None
        
        
    def start(self, token):
        self.bot = telepot.aio.DelegatorBot(token, [
            include_callback_query_chat_id(
                pave_event_space())(
                per_chat_id(), create_open, SarcasmUser, timeout=10),
            ])
        
        self.msg_loop = MessageLoop(self.bot)
        self.loop.create_task(MessageLoop(self.bot).run_forever())
        self.loop.run_forever()
        
    def stop(self):
        SarcasmBot.instance = None
        if self.msg_loop:
            self.msg_loop.cancel()
            self.msg_loop = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, tb):
        self.stop()



        
class SarcasmUser(telepot.aio.helper.ChatHandler):
    
    def __init__(self, *args, **kwargs):
        super(SarcasmUser, self).__init__(*args, **kwargs)
        print('Created {}'.format(self.id))
       
    
    async def on_chat_message(self, msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        
        if 'text' in msg:
            if msg['text'] == '/help':
                await self.sender.sendMessage("Send a message. If i think it's sarcastic i will send a clown emoji, else i will show an OK emoji")
                return
            
            message = msg['text']
            print(message)
            
            trans_dict = datasetExtractor.convertToLTEC(message)
            trans_mess = classifier.transformData(predictor, trans_dict)
            prediction = predictor["logit"].predict(trans_mess)[0]


            if prediction:
                # is sarcastic
                await self.sender.sendMessage("🤡")
            else:
                # isn't sarcastic
                await self.sender.sendMessage("👌")

                
        else:
            await self.sender.sendMessage("Message must be a text")
            
    async def on_close(self, ex):
        print('Closed {}'.format(self.id))



if __name__ == '__main__':


    path1 = "dataset\\model\\Lemmatized_tokenized_emote_counted_model_2.pkl"
    path2 = "data\\lemmatized_tokenized_emote_counted.pkl"

    try:
        # best performing model in the tests
        predictor = classifier.loadModel(path1)

    except IOError:
        df = classifier.loadModel(path2)
        x_train, x_test, y_train, y_test = train_test_split(df[df.columns[~df.columns.isin(['subreddit','author','label'])]], df['label'], random_state=10)

        predictor = classifier.modelFitting(x_train, y_train)
        
        saveModel(path1,predictor)    

    print("Bot started") 
    # Se crea un bot i inicia
    with SarcasmBot() as bot:

        # Start bot
        bot.start(open('TOKEN').read().strip())