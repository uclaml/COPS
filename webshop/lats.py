
#!/usr/bin/env python
# coding: utf-8

# # Setup

# In[1]:

import os
import sys
import copy
import itertools
import numpy as np
from functools import partial
from models import gpt
import requests
import logging
import random
 
completion_tokens = prompt_tokens = 0

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment

WEBSHOP_URL = "http://127.0.0.1:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
  return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")


def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )


def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', **kwargs):
    if page_type == 'init':
      url = (
          f'{WEBSHOP_URL}/{session}'
      )
    if page_type == 'search':
      url = (
          f'{WEBSHOP_URL}/search_results/{session}/'
          f'{query_string}/{page_num}'
      )
    elif page_type == 'item':
      url = (
          f'{WEBSHOP_URL}/item_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{options}'
      )
    elif page_type == 'item_sub':
      url = (
          f'{WEBSHOP_URL}/item_sub_page/{session}/'
          f'{asin}/{query_string}/{page_num}/{subpage}/{options}'
      )
    elif page_type == 'end':
      url = (
          f'{WEBSHOP_URL}/done/{session}/'
          f'{asin}/{options}'
      )
    # print(url)
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    # visible_texts = [str(text).strip().strip('\\n') for text in visible_texts]
    # if page_type == 'end': import pdb; pdb.set_trace()
    if False:
        # For `simple` mode, return just [SEP] separators
        return ' [SEP] '.join(t.strip() for t in visible_texts if t != '\n')
    else:
        # Otherwise, return an observation with tags mapped to specific, unique separators
        observation = ''
        option_type = ''
        options = {}
        asins = []
        cnt = 0
        prod_cnt = 0
        just_prod = 0
        for t in visible_texts:
            if t == '\n': continue
            if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': continue
            # if t.startswith('Instruction:') and page_type != 'init': continue
            # print(t.parent.name, t)
            if t.parent.name == 'button':  # button
                processed_t = f'\n[{t}] '
            elif t.parent.name == 'label':  # options
                if f"'{t}'" in url:
                    processed_t = f'[[{t}]]'
                    # observation = f'You have clicked {t}.\n' + observation
                else:
                    processed_t = f'[{t}]'
                options[str(t)] = option_type
                # options[option_type] = options.get(option_type, []) + [str(t)]
            elif t.parent.get('class') == ["product-link"]: # product asins
                processed_t = f'\n[{t}] '
                if prod_cnt >= 10:
                  processed_t = ''
                prod_cnt += 1
                asins.append(str(t))
                just_prod = 0
            else: # regular, unclickable text
                processed_t =  '\n' + str(t) + ' '
                if cnt < 2 and page_type != 'init': processed_t = ''
                if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
                option_type = str(t)
                cnt += 1
            just_prod += 1
            observation += processed_t
        info = {}
        if options:
          info['option_types'] = options
        if asins:
          info['asins'] = asins
        if 'Your score (min 0.0, max 1.0)' in visible_texts:
          idx = visible_texts.index('Your score (min 0.0, max 1.0)')
          info['reward'] = float(visible_texts[idx + 1])
          observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
        return clean_str(observation), info

class webshopEnv:
  def __init__(self):
    self.sessions = {}

  def clone_state(self):
    return copy.deepcopy(self.sessions)
  
  def step(self, session, action):
    done = False
    observation_ = None
    logging.info(self.sessions)
    if action == 'reset':
      self.sessions[session] = {'session': session, 'page_type': 'init'}
    elif action.startswith('think['):
      observation = 'OK.'
    elif action.startswith('search['):
      assert self.sessions[session]['page_type'] == 'init'
      query = action[7:-1]
      self.sessions[session] = {'session': session, 'page_type': 'search',
                                'query_string': query, 'page_num': 1}
    elif action.startswith('click['):
      button = action[6:-1]
      if button == 'Buy Now':
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'end'
        #done = True
      elif button == 'Back to Search':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        self.sessions[session] = {'session': session, 'page_type': 'init'}
      elif button == 'Next >':
        #assert False # ad hoc page limitation
        assert self.sessions[session]['page_type'] == 'search'
        self.sessions[session]['page_num'] += 1
      elif button == '< Prev':
        assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
        if self.sessions[session]['page_type'] == 'search':
          #assert False
          self.sessions[session]['page_num'] -= 1
        elif self.sessions[session]['page_type'] == 'item_sub':
          self.sessions[session]['page_type'] = 'item'
        elif self.sessions[session]['page_type'] == 'item':
          self.sessions[session]['page_type'] = 'search'
          self.sessions[session]['options'] = {}
      elif button in ACTION_TO_TEMPLATE:
        assert self.sessions[session]['page_type'] == 'item'
        self.sessions[session]['page_type'] = 'item_sub'
        self.sessions[session]['subpage'] = button
      else:
        if self.sessions[session]['page_type'] == 'search':
          assert button in self.sessions[session].get('asins', [])  # must be asins
          self.sessions[session]['page_type'] = 'item'
          self.sessions[session]['asin'] = button
        elif self.sessions[session]['page_type'] == 'item':
          assert 'option_types' in self.sessions[session]
          assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])  # must be options
          option_type = self.sessions[session]['option_types'][button]
          if not 'options' in self.sessions[session]:
            self.sessions[session]['options'] = {}
          self.sessions[session]['options'][option_type] = button
          observation_ = f'You have clicked {button}.'
    else:
      assert False
    observation, info = webshop_text(**self.sessions[session])
    if observation_:
      observation = observation_
    self.sessions[session].update(info)
    reward = info.get('reward', 0.0)
    if reward != 0.0:
        #print(f"Current Session State: {self.sessions[session]}")
        #print(f"Action being processed: {action}")
        print(f"Resulting Observation: {observation}")
        observation+=" Please try again!"
    if reward == 1.0:
        done = True
        print("done")
    return observation, reward, done

env = webshopEnv()

logging.info("Logging has been configured.")

global reflection_map
reflection_map = []


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1



def get_samples(task, x, y, n_generate_sample, prompt_sample, knn=None):
    unique_trajectories = get_unique_trajectories(failed_trajectories)
    global reflection_map
    global completion_tokens
    global prompt_tokens
    reflection_map = []
    if prompt_sample == "standard":
        prompt = task.standard_prompt_wrap(x, y, [])
    elif prompt_sample == "cot":
        prompt = task.cot_prompt_wrap(x, y, [], knn)
    else:
        raise ValueError(f"prompt_sample {prompt_sample} not recognized")
    logging.info(f"PROMPT: {prompt}")
    samples,inca,incb = gpt(prompt, n=n_generate_sample)
    prompt_tokens+=inca
    completion_tokens+=incb
    #for trt in samples:
    #    with open("debug_log1.log", "a", encoding="utf-8") as f:
    #        f.write(y+"\n"+trt+"\n")
    return [y + _ for _ in samples]


def get_unique_trajectories(failed_trajectories, num=2):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get("final_answer")
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj["trajectory"]))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories


class Node:
    def __init__(self, state, question, parent=None, knn=None,env_state=None):
        self.state = (
            {"action": "", "observation": ""} if state is None else state
        )
        self.parent = parent
        self.question = question
        self.children = []
        self.visits = 0
        self.value = 0
        self.depth = 0 if parent is None else parent.depth + 1
        self.is_terminal = False
        self.reward = 0
        self.exhausted = False  # If all children are terminal
        self.em = 0  # Exact match, evaluation metric
        self.knn = knn
        self.env_state = env_state

    def ques(self):
        return self.question

    def uct(self):
        if self.visits == 0:
            # return float('inf')
            return self.value * 2
        return self.value / self.visits + np.sqrt(
            2 * np.log(self.parent.visits) / self.visits
        )

    def uct_with_depth(self, C1=1, C2=1):
        if self.visits == 0:
            return self.value
        exploitation_term = self.value / self.visits
        exploration_term = np.sqrt(2 * np.log(self.parent.visits) / self.visits)
        depth_term = self.depth
        return exploitation_term + C1 * exploration_term + C2 * depth_term

    def __str__(self):
        return f"Node(depth={self.depth}, value={self.value:.2f}, visits={self.visits}, action={self.state['action']}, observation={self.state['observation']}<end_of_obs>)"

    def to_dict(self):
        return {
            "state": self.state,
            "question": self.question,
            "parent": self.parent.to_dict() if self.parent else None,
            "children": [child.to_dict() for child in self.children],
            "visits": self.visits,
            "value": self.value,
            "depth": self.depth,
            "is_terminal": self.is_terminal,
            "reward": self.reward,
            "em": self.em,
            "knn": self.knn,
        }


def node_trajectory_to_text(node_string):
    lines = node_string.split("Node(")
    formatted_lines = []
    for line in lines:
        if line.startswith("Instruction"):
            formatted_lines.append(line)
            continue
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split("<end_of_obs>)")[0].strip()
        except IndexError:
            continue

        if depth != 0:
            if action:
                formatted_lines.append(f"Action: {action}")
            if observation:
                formatted_lines.append(f"Observation: {observation}")
    formatted_lines.pop()
    return "\n".join(formatted_lines)


def traj_depth(node_string):
    lines = node_string.split("\n")
    formatted_lines = []
    ret = 0
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split("<end_of_obs>)")[0].strip()
        except IndexError:
            continue
        if depth > ret:
            ret = depth
    return ret


def collect_all_nodes(node):
    """Recursively collect all nodes starting from the given node."""
    nodes = [node]
    for child in node.children:
        nodes.extend(collect_all_nodes(child))
    return nodes


def collect_trajectory(node):
    trajectory = []
    ques = ""
    while node:
        ques = "Instruction: " + str(node.question).replace("WebShop","").replace("Instruction:","").strip()
        trajectory.append(str(node))
        node = node.parent
    if len(ques) > 0:
        trajectory.append(ques)
    return "\n".join(reversed(trajectory))

import re
def get_substrings_between_brackets(s):
    pattern = r"\[(.*?)\]"
    matches = re.findall(pattern, s)
    return matches[0]

def quote_env(idx):
    return str(env.step(idx, "reset")[0])


def dfs_search(args, task, idx, iterations, knnret, depth_limit=30, to_print=True):
    global gpt
    global failed_trajectories
    global success_trajectories
    gpt = partial(gpt, model_size=args.model_size, temperature=args.temperature)
    action="reset"
    x = env.step(idx, action)[0]
    if to_print:
        print(idx, x)
    root = Node(state=None, question=x)
    root.env_state = copy.deepcopy(env.sessions)
    all_nodes = []
    failed_trajectories = []
    success_trajectories = []
    stack = [root]
    it = 0
    knn = []
    if knnret:
        for traj in knnret:
            format_traj = node_trajectory_to_text(traj["trajectory"])
            # format_traj+=f"Action {traj_depth(traj['trajectory'])}: Finish[{get_substrings_between_brackets(traj['final_answer'])}]"+"\n"
            knn.append(format_traj)
        print("--------------knn is:")
        print(knn)
    last_node = None
    maxx=0.0
    while stack and it < iterations:
        node = stack.pop()
        last_node = node
        logging.info(f"DFS at node depth {node.depth}...")

        if node.is_terminal and node.reward == 1:
            logging.info(f"Terminal node with reward 1 found at depth {node.depth}")
            return (
                node.state,
                node.value,
                all_nodes,
                node.reward,
                node.em,
                failed_trajectories,
                success_trajectories,
            )

        if node.is_terminal and node.reward == 0:
            logging.info(f"Terminal node with reward 0 found at depth {node.depth}")
            return (
                node.state,
                node.value,
                all_nodes,
                maxx,
                node.em,
                failed_trajectories,
                success_trajectories,
            )
            
        maxx=max(node.reward,maxx)
        
        if node.depth >= depth_limit:
            logging.info("Depth limit reached")
            it += 1
            continue  # go to next iteration

        expand_node(node, args, task,idx, knn=knn)
        stack.extend(reversed(node.children))  # adding all child nodes to stack for DFS

        all_nodes = [(node, node.value) for node in collect_all_nodes(root)]
        logging.info(f"State of all_nodes after iteration: {all_nodes}")
        it += 1
    # If we reach here, no solution was found
    logging.info("All paths explored. No solution found.")
    if len(failed_trajectories) == 0:
        trajectory = collect_trajectory(last_node)
        failed_trajectories.append({"trajectory": trajectory, "final_answer": ""})
    return root, 0, all_nodes, maxx, 0, failed_trajectories, success_trajectories


def select_node_dfs(stack):
    return stack[-1] if stack else None  # return the last node in the stack




def select_node(node):
    while node and node.children:
        logging.info(
            f"Selecting from {len(node.children)} children at depth {node.depth}."
        )

        terminal_children = [child for child in node.children if child.is_terminal]
        terminal_status = [child.is_terminal for child in node.children]

        if len(terminal_children) == len(node.children):
            logging.info(
                f"All children are terminal at depth {node.depth}. Backtracking..."
            )
            if node.parent:
                node.parent.children.remove(node)
            node = node.parent
            continue

        node_with_reward_1 = next(
            (child for child in terminal_children if child.reward == 1), None
        )
        if node_with_reward_1:
            logging.info(f"Found terminal node with reward 1 at depth {node.depth}.")
            return node_with_reward_1

        node = max(
            (child for child in node.children if not child.is_terminal),
            key=lambda child: child.uct(),
            default=None,
        )

        while node.is_terminal and node.reward != 1:
            node = max(
                (child for child in node.parent.children if not child.is_terminal),
                key=lambda child: child.uct(),
                default=None,
            )

        logging.info(f"Selected node at depth {node.depth} with UCT {node.uct()}.")

    return node  # This will return None if all paths from the root are exhausted


def expand_node(node, args, task, idx,knn=None):
    if node.depth >= 30:
        logging.info("Depth limit reached")
        print("Depth limit reached")
        node.is_terminal = True
        return
    new_nodes = generate_new_states(node, args, task, idx,knn=knn)
    node.children.extend(new_nodes)


def generate_new_states(node, args, task, idx, knn=None):
    prompt = generate_prompt(node)
    #print("the prompt is:")
    #print(prompt)
    sampled_actions = get_samples(
        task,
        prompt,
        f"Action: ",
        args.n_generate_sample,
        prompt_sample=args.prompt_sample,
        knn=knn,
    )
    logging.info(f"SAMPLED ACTION: {sampled_actions}")

    unique_states = {}  # Store unique states here
    for action in sampled_actions:
        local_sessions = copy.deepcopy(node.env_state)
        env.sessions = local_sessions
        new_state = node.state.copy()  # Make a copy of the parent node's state

        
        action_line = next(
            (
                line.split(":")[1].strip()
                for line in action.split("\n")
                if line.startswith("Action") and ":" in line
            ),
            None,
        )

        # Use thought and action to form a unique key
        unique_key = f"{action_line}"

        if unique_key in unique_states:
            continue  # Skip if this state already exists

        if action_line:
            #print("the action line is:")
            #print(action_line)
            try:
                res = env.step(idx, action_line)
                #print("res", res)
                obs = res[0]
                r = res[1]
                done = res[2]
            except AssertionError:
                obs = 'Invalid action!'
                # print("err")
                r = -1
                done = False

            if action.startswith('think'):
                observation = 'OK.'

            # Update the new state dictionary
            new_state["action"] = action_line
            new_state["observation"] = obs

            env_state_clone = env.clone_state() 
            new_node = Node(state=new_state, question=node.question,  env_state=env_state_clone,parent=node)
            new_node.env_state = local_sessions
            new_node.is_terminal = r == 1 or done
            new_node.reward = r
            unique_states[unique_key] = new_node  # Add this state to unique_states
            logging.info(f"NEW NODE: {new_node}")

            if new_node.is_terminal and r == 0:
                trajectory = collect_trajectory(new_node)
                failed_trajectories.append(
                    {
                        "trajectory": trajectory,
                        "final_answer": f"{action_line}",
                    }
                )
            if new_node.is_terminal and r == 1:
                trajectory = collect_trajectory(new_node)
                success_trajectories.append(
                    {
                        "trajectory": trajectory,
                        "final_answer": f"{action_line}",
                    }
                )

    return list(unique_states.values())  # Return unique nodes as a list




def print_tree(node, level=0):
    indent = "  " * level
    print(f"{indent}{node}")
    for child in node.children:
        print_tree(child, level + 1)




def generate_prompt(node):
    trajectory = []
    question = node.question
    while node:
        new_segment = []
        if node.state["action"]:
            new_segment.append(f"Action: {node.state['action']}")
        if (
            node.state["observation"] and node.depth != 0
        ):  # Exclude the observation from the root node
            new_segment.append(f"Observation: {node.state['observation']}")
        trajectory.append("\n".join(new_segment))
        node = node.parent
    return question + "\n".join(reversed(trajectory))
