# Writing Custom Surfkit-based Agents

OSUniverse relies on the [Surfkit](https://github.com/agentsea/surfkit) framework, which allows creating containerized vistual desktops, trackers and agents. 

However, using Surfkit is not mandatory. If you want to write your own agent with your own virtual device, please refer to [Writing Custom Agent Runners](RUNNERS.md).

## Writing Custom Agents

To write your own Surfkit-based agent, you can start from any existing agent in the `agents` directory. If you take, for example, the `react` agent as a base, you will need to re-implement the `Actor` class in the `actor/oai.py` file. You can also rewrite the `agent.py` file, but in most cases this is not required. 

The main `Agent` class and the `Actor` class communicate with each other through the `act` method. The `act` method should return a `Step` object, which will be executed by the device in the main loop. The `act` method accepts the following arguments:

* `task`: a `Task` object, which contains the task description and other metadata.
* `device`: a `Desktop` object, which is a Surfkit-based virtual device.
* `history`: a list of `Step` objects, which contains the history of the agent's actions.

The role of the actor is to communicate with the LLMs and to generate the next action to be executed by the device. The `act` method should return a `Step` object, which contains the action to be executed by the device. This object should also contain extra information, that helps to measure the agent performance and to calculate the statistics.

```python
step = Step(
    state=EnvState(images=screenshots), # the current state of the environment
    action=action, # the action to be executed by the device, of type `V1Action`    
    thought=thought, # the agent's thought, string: not required, 
                        # but recommended for the human reviewer convinience
    raw_response=completion_response.choices[0].message.content, # the raw response from the LLM, 
                                                                    # used for the same purpose as the `thought` field
    task=task, # the task object, the same as the argument of the `act` method
    thread=thread, # the thread object, optional; should be used 
                    # in case of the communication with the LLM is done with a ThreadMem framework
    model_id=self.model, # the model ID, string
    in_tokens=completion_response.usage.prompt_tokens, # the number of input tokens used by the LLM
    out_tokens=completion_response.usage.completion_tokens, # the number of output tokens used by the LLM
)
```

If you don't modify the `Agent` class, it takes care of the loop and the control over the maximum number of steps.

## Custom Action Space

A lot of models have a custom action space they were trained on, whereas Surfkit-based agents should run a set of actions specific to the [AgentDesk](https://github.com/agentsea/agentdesk) framework.

To make the agent work with a custom action space, you should write a convertor. You can see the examples of the convertors in the `agents/cua/cua/actor/action_parser.py` and `agents/qwen/qwen/actor/action_parser.py` files.

## Dependencies

Each agent has its own set of dependencies. Each agent is a Poetry project, so you should install the dependencies and the agent with the following command:

```bash
poetry install
```

## Running the Benchmark with a Custom Agent

Note: the runner is passing a couple of arguments to the agent in a form of environment variables:

```python
os.environ["SURFKIT_AGENT_MODEL"] = config.agent_model
os.environ["SURFKIT_AGENT_MODEL_BASE_URL"] = config.agent_model_base_url
```

You can use these variables inside your agent.

Running the benchmark with a custom agent is done in the same way, as with the default agents we provide. See [README.md](../README.md) for more information.
