import os
import time
import streamlit as st
from query import *
from upload import *

def clear_history():
    pass

def site():
    # st.image('img.png')
    #subheader 
    st.subheader('LLM Question-Answering Application 👋')

    #sidebar creator
    with st.sidebar:
        #OpenAI API KEY
        api_key = st.text_input('OpenAI API Key:', type='password')
        
        #environment variable
        if api_key:
            os.environ['OPENAI_API_KEY'] = api_key

        #uploads file    
        uploaded_file = st.file_uploader('Upload a file:', type=('pdf', 'docx', 'txt'))

        #number input widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        #number input widget
        overlap = st.number_input('Overlap Ratio:', min_value=0.1, max_value=0.99, value=0.25, on_change=clear_history)

        #variable k input
        top_k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        #add data button
        add_data = st.button('Add Data', on_click=clear_history)

        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):
                # writing the file from RAM to the current directory on disk
                time.sleep(5)
                bytes_data = uploaded_file.read() 
                file_name = os.path.join('./Docs/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                pc = create_pinecone_index(api_key, index_name)
                #read_pdf_file("Attention_is_all_you_need.pdf")
                nodes = splitter_function(read_pdf_file(uploaded_file.name), chunk_size, overlap)
                embeddings = create_embeddings(nodes)
                upload_to_pinecone(pc, index_name, embeddings, nodes)

            st.write("Got PDF")
            st.write(uploaded_file)
            st.write(chunk_size)
            st.write(top_k)


    query = st.text_input("Query: ")

    if query:
        print("Question: " + query + " top k:" + str(top_k))


        pc = Pinecone(api_key=api_key)

        pc_index = pc.Index(index_name)


        # Create embeddings for the query
        Settings.embed_model = embed_model
        query_embedding = embed_model.get_text_embedding(query)
        print(query_embedding)

        # Get uery result from Pinecone
        model_url= "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"
        llm = LlamaCPP(
            model_url=model_url,
            # optionally, you can set the path to a pre-downloaded model instead of model_url
            #model_path="./llm/llama-2-7b-chat.Q2_K.gguf",
            temperature=0.1,
            max_new_tokens=384,
            context_window=3000,
            generate_kwargs={},
            verbose=False,) 
        
        
        query_result = pc_index.query(vector = query_embedding, top_k=top_k, include_values=True, include_metadata=True)


        _nodes= []
        print("Query: ", query)
        print("Retrieval")
        print("top-k :", top_k)
        for i, _t in enumerate(query_result['matches']):
            print(i, end=':  ')
            # print(_t)
            _node =IndexNode.from_text_node(TextNode(text=_t['metadata']['text']), _t['id'])
            _nodes.append(_node)
            print("Similarity: ", _t['score'])
            print("---------------------------")
            
        print(_nodes)

        if _nodes is not None and len(_nodes) >= 1 and isinstance(_nodes[0], BaseNode):
            print("----------------------------ENTER --------------------------")
        # create vector store index
        _index = VectorStoreIndex(_nodes)

        #Re-rank
        query_engine = _index.as_query_engine(similarity_top_k=top_k, llm=llm)
        response = query_engine.query(query)
        print("Answer: ")
        print(str(response))

        st.write(str(response))



if __name__ == '__main__':
    # try:
    site()
    # except:
    #     st.write("# OOPs. Something went wrong check the parameters in the sidebar")