'''
Prompt the model for knowledge self-exploration
'''
knowpath_prompt = """
You need to answer Question using follow steps:
step1:You need to extract the most relevant topic entities from the Question.\n
step2:Based on the topic entities and Question. List the 15 related knowledge triples from high to low in terms of relevance to the Question . The triples are given in the form of (entity, relation, entity).\n
step3:Based on the knowledge triples you listed, combined with the Question and topic entities, you need to give the final answer. In addition, you need to give the reasoning path. The overall format should be "entity1->relation1->entity2->relation2->entity3->...->end".\n
The answer format is {reasoning_path : ["entity1->relation1->entity2->relation2->entity3->...->end"], "response": "based on the knowledge, the answer to the question $question is xxxx" }\n

Question: $question.\n
Answer:\n

"""


'''
The prompt for exploring the most relevant relationships can investigate several related connections.
'''
knowpath_prune_relation_prompt = """
Dict : {
"Question" : $question,
"Topic enetity" : $topicEntity,
}
RelationList: $relationList\n
Now you need to find out up to 7 most relevant relations from RelationList to each entry in the dictionary Dict and put them into a list called Relations. The answer format is: {"Relations":[xxx, xxx, xxx,...] (length up to 5)}. Do not output any extra content except what is required by the format. \n

Answer:\n
"""

knowpath_prune_relation_prompt_with_str = """
Dict : {
"Question" : $question,
"Topic enetity" : $topicEntity,
"Knowledge Path" : $knowpath_str,
}
RelationList: $relationList\n
Now you need to find out up to 7 most relevant relations from RelationList to each entry in the dictionary Dict and put them into a list called Relations. The answer format is: {"Relations":[xxx, xxx, xxx,...] (length up to 5)}. Do not output any extra content except what is required by the format. \n

Answer:\n
"""

'''
exploring the most relevant entities can investigate several related entities.
'''
knowpath_prune_entity_prompt = """
Dict : {
"Question" : $question,
"Topic enetity" : $topicEntity,
"RelationList" : $relationList,
}
EntityList: $entityList\n
Now you need to find out up to 7 entities that are most relevant to each entry in the dictionary Dict from EntityList by relevance, and put them into a list called Entities. The answer format is: {"Entities":[xxx, xxx, xxx,...] (length up to 5)}. Do not output any extra content except what is required by the format. \n
Answer:\n
"""


knowpath_prune_entity_prompt_with_str = """
Dict : {
"Question" : $question,
"Topic enetity" : $topicEntity,
"Knowledge Path" : $knowpath_str,
"RelationList" : $relationList,
}
EntityList: $entityList\n
Now you need to find out up to 7 entities that are most relevant to each entry in the dictionary Dict from EntityList by relevance, and put them into a list called Entities. The answer format is: {"Entities":[xxx, xxx, xxx,...] (length up to 5)}. Do not output any extra content except what is required by the format. \n

Answer:\n
"""


'''
evaluation
'''
knowpath_evaluation_prompt = """
Reasoning_path:$subgraph
Based on the Reasoning_path and your own knowledge, you need to determine whether the Question:$question can be answered. '->' and '<-' indicate the direction of Reasoning_path between entities and relationships.\n
Requests:\n
1.The answer format is: {"Answerable": True or False,"Response": "the answer to the question $question is xxxx"}\n

Answer: \n

"""


'''
COT method
'''
cot_prompt = """

Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

Q:$question
A:
"""

