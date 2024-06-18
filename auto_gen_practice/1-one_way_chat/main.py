import autogen 


def main(): 

    config_list = autogen.config_list_from_json(
        env_or_file="config_list.json", 
        filter_dict={
        "model": ["gemini-1.0-pro"],
        },
    )

    assistant = autogen.AssistantAgent(
        name="awesme_assistant", 
        llm_config={
            "config_list": config_list
            
        }
    )

    user_proxy = autogen.UserProxyAgent(
        name="user", 
        human_input_mode="NEVER", 
        code_execution_config={
            "work_dir": "coding", 
            "use_docker": False
        }
    )

    user_proxy.initiate_chat(assistant, message="write a english poem in python")



if __name__ == "__main__": 
    main()