from SPARQLWrapper import SPARQLWrapper, JSON
from utils_knowpath import *
import json
import time
import openai
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
import copy

SPARQLPATH = "http://localhost:8899/sparql"  # depend on your own internal address and port

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""

def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def fix_sparql_query(query):
    """
    Fix namespace concatenation errors in SPARQL query.
    """
    import re
    pattern = r'ns:(http[^\s]*)'

    if re.search(pattern, query):
        return re.sub(pattern, r'<\1>', query)
    else:
        return query

def execurte_sparql(sparql_query):
    sparql_query = fix_sparql_query(sparql_query)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def clean_relations(string, entity_id, head_relations):
    pattern = r"{\s*(?P<relation>[^()]+)\s+\(Score:\s+(?P<score>[0-9.]+)\)}"
    relations=[]
    for match in re.finditer(pattern, string):
        relation = match.group("relation").strip()
        if ';' in relation:
            continue
        score = match.group("score")
        if not relation or not score:
            return False, "output uncompleted.."
        try:
            score = float(score)
        except ValueError:
            return False, "Invalid score"
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": score, "head": False})
    if not relations:
        return False, "No relations found"
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def dedupe_lists1(lst):
    return [list(x) for x in set(tuple(x) for x in lst)]

class SubGraphExploration():
    """
    Subgraph exploration classes
    """

    def __init__(self, topic_entity_id, args):
        self.sparql_relation_as_head = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?relation
        WHERE {
            ns:%s ?relation ?tail .
        }
        """
        self.sparql_relation_as_tail = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT DISTINCT ?relation
        WHERE {
            ?head ?relation ns:%s .
        }
        """
        self.sparql_entity_as_head = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?Entity
        WHERE {
        ns:%s ns:%s ?Entity .
        }
        """
        self.sparql_entity_as_tail = """
        PREFIX ns: <http://rdf.freebase.com/ns/>
        SELECT ?Entity
        WHERE {
        ?Entity ns:%s ns:%s .
        }
        """

        self.resoning_path = [[] for item in range(3)]
        self.topic_entity_id = topic_entity_id
        self.args = args

    def id2entity_name_or_type_new(self, entity_id):
        """Retrieve entity names by ID

        Args:
            entity_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        try:
            sparql_query = sparql_id % (entity_id, entity_id)
            sparql = SPARQLWrapper(SPARQLPATH)
            sparql.setQuery(sparql_query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            if len(results["results"]["bindings"])==0:
                return "UnName_Entity"
            else:
                for item in results["results"]["bindings"]:
                    if item["tailEntity"]["xml:lang"] == "en":
                        return item['tailEntity']['value']
        except:
            pass
        
    def write_list_to_json(self, data_list, filename):
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json_str = json.dumps(data_list, ensure_ascii=False)
            f.write(json_str + '\n')
            
    def limit_list_size(self, lst, limit=40):
        """
        If list exceeds limit, randomly sample specified number of elements
        """
        if len(lst) > limit:
            return random.sample(lst, limit)
        return lst

    def filter_relation(self, item):
        """Filter out obviously useless knowledge triples

        Args:
            item (_type_): _description_

        Returns:
            _type_: _description_
        """
        flag = False
        if "http://www.w3.org" in item["relation"]["value"]:
            flag = True
        elif "type.object.name" in item["relation"]["value"]:
            flag = True
        elif "http://rdf.freebase.com/key" in item["relation"]["value"]:
            flag = True
        elif "common.topic.topic_equivalent_webpage" in item["relation"]["value"]:
            flag = True
        elif "type.object.key" in item["relation"]["value"]:
            flag = True
        elif "topic_server.population_number" in item["relation"]["value"]:
            flag = True
        elif "type.object.type" in item["relation"]["value"]:
            flag = True
        elif "common.topic.description" in item["relation"]["value"]:
            flag = True
        else:
            flag = False

        return flag
    
    def add_dicts(self, dict1, dict2):
        """Merge keys from both dictionaries

        Args:
            dict1 (_type_): _description_
            dict2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        result = {key: dict1.get(key, 0) + dict2.get(key, 0) for key in dict1}
        
        return result
    def get_most_relevant_relations(self, relation_list, entity_name, question, args, knowpath_str=''):
        """Find most relevant relations based on the relation list

        Returns:
            _type_: _description_
        """
        if args.method == 'knowpath_wo_p':
            get_relevant_relations_prompt = Template(knowpath_prune_relation_prompt).substitute(
                        question=question, 
                        topicEntity=entity_name,
                        relationList = relation_list)
        else:
            get_relevant_relations_prompt = Template(knowpath_prune_relation_prompt_with_str).substitute(
                        question=question, 
                        topicEntity=entity_name,
                        relationList = relation_list,
                        knowpath_str = knowpath_str)
        response, token_num = run_ollama(args, get_relevant_relations_prompt, 
                            args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        self.args.result_dict["call_num"] += 1
        if 'token_num' not in self.args:
            self.args.result_dict['token_num'] = token_num
        else:
            self.args.result_dict['token_num'] = self.add_dicts(self.args.result_dict['token_num'], token_num)

        return response
    
    def get_most_relevant_entities(self, entity_list, entity_name, question, args, most_relevant_relations_list='', knowpath_str=''):
        """Find most relevant entities
        
        Returns:
            _type_: _description_
        """
        
        if args.method == 'knowpath_wo_p':
            get_relevant_entities_prompt = Template(knowpath_prune_entity_prompt).substitute(
                question=question, 
                topicEntity=entity_name,
                entityList = entity_list,
                relationList = most_relevant_relations_list)
        else:
            get_relevant_entities_prompt = Template(knowpath_prune_entity_prompt_with_str).substitute(
                question=question, 
                topicEntity=entity_name,
                entityList = entity_list,
                relationList = most_relevant_relations_list,
                knowpath_str = knowpath_str)
        response, token_num = run_ollama(args, get_relevant_entities_prompt, 
                            args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type)
        self.args.result_dict["call_num"] += 1
        self.args.result_dict['token_num'] = self.add_dicts(self.args.result_dict['token_num'], token_num)

        return response

    def find_relation(self, sparql_relation_sql, entity_id):
        """Query entity relations using SPARQL
        """
        relation_sql = sparql_relation_sql % (entity_id)
        relations = execurte_sparql(relation_sql)
        relation_list = []
        for id, item in enumerate(relations):
            try:
                flag = self.filter_relation(item)
                if not flag:
                    relation_list.append(item["relation"]["value"])
            except:
                continue
        relation_list = list(set(relation_list))
        relation_list = [item.replace("http://rdf.freebase.com/ns/","") for item in relation_list]
        
        return relation_list

    def find_enetity(self, entity_id, relation, head=True):
        """Query entities mapped by relations via SPARQL
        """
        if head:
            sprql = self.sparql_entity_as_head
            enetity_sql = sprql % (entity_id, relation)
        else:
            sprql = self.sparql_entity_as_tail
            enetity_sql = sprql % (relation, entity_id)
        eneity_list = []
        entities = execurte_sparql(enetity_sql)
        for id, item in enumerate(entities):
            try:
                eneity_list.append(item["Entity"]["value"])
            except:
                continue
        eneity_list = list(set(eneity_list))

        return eneity_list

    def subtract_lists_2(self, list1, list2):
        return list(set(list1) - set(list2))
    
    
    def extract_json(self, text):
        pattern = r'{.*}'
        match = re.search(pattern, text)
        if match:
            json_str = match.group()
            return json.loads(json_str)
        return None
    
    def extract_list_from_string(self, input_str):
        try:
            if not input_str:
                return []
                
            if not isinstance(input_str, str):
                return []
                
            start = input_str.find('[')
            end = input_str.rfind(']')
            
            if start == -1 or end == -1:
                return []
                
            content = input_str[start + 1:end]
            
            if not content.strip():
                return []
                
            items = content.split(',')
            
            result = []
            for item in items:
                cleaned = item.strip().strip('"').strip("'")
                if cleaned: 
                    result.append(cleaned)
                    
            return result
            
        except Exception:
            return []
        

    def flatten_list(self, nested_list):
        return [item for sublist in nested_list if sublist for item in sublist]
    

    def subgraph_exploreration_more(self, enetity_dict, entity_name, question, args, 
                                    depth='', path = False, knowpath_str=''):
        print(f"\nCurrent exploration depth:{depth}")
        if depth == 0:
            path = [[entity_name[0]] for i in range(args.max_entity_width)]

        
        original_path = copy.deepcopy(path)
        all_candidate_entities_group_list = [[] for item in range(len(enetity_dict))]
        head_or_tail_list_group = [[] for item in range(len(enetity_dict))]
        all_candidate_entities_name_group = [[] for item in range(len(enetity_dict))]
        most_relevant_relations_list_group = [[] for item in range(len(enetity_dict))]
        rel_entity_dict_list = []
        entity_name_id_dict = {}
        for entity_idx, eid in enumerate(enetity_dict):
            topic_entity = entity_name[entity_idx]
            rel_entity_dict =  {}
            print(f"\nEntity: {topic_entity}, Question：{question}")
            as_head_relation_list = self.find_relation(self.sparql_relation_as_head, eid)
            as_tail_relation_list = self.find_relation(self.sparql_relation_as_tail, eid)
            total_rel_list = as_head_relation_list + as_tail_relation_list
            most_relevant_relations = self.get_most_relevant_relations(total_rel_list, 
                                                                   topic_entity, question, args, knowpath_str)
            most_relevant_relations_list = self.extract_list_from_string(most_relevant_relations)
            print(f"Most related relations: {most_relevant_relations_list}")
            most_relevant_relations_list_group[entity_idx].extend(most_relevant_relations_list)
            head_or_tail_list_group[entity_idx] = [True]*len(most_relevant_relations_list)
            for relation_id, relation in enumerate(most_relevant_relations_list):
                rel_entity_dict[relation] = ''
                if relation in head_or_tail_list_group[entity_idx]:
                    pass
                else:
                    head_or_tail_list_group[entity_idx][relation_id] = False
            candidate_entity_list = [[] for item in range(len(most_relevant_relations_list))]
            all_candidate_entities = []
            all_candidate_entities_name  = []
            for id, relation in enumerate(most_relevant_relations_list):
                if head_or_tail_list_group[entity_idx][id]:
                    entities = self.find_enetity(eid, relation)
                    entities = list(set(self.limit_list_size(entities)))
                    entities = [item.replace("http://rdf.freebase.com/ns/","") for item in entities]
                    entities = [entity for entity in entities if entity.startswith("m.")]
                    candidate_entity_list[id] = entities
                    entities_name = [self.id2entity_name_or_type_new(item) for item in entities]
                    all_candidate_entities.extend(entities)
                    all_candidate_entities_name.extend(entities_name)
                    entity_name_id_dict = self.set_key_value(entities_name, entities, entity_name_id_dict)
                    rel_entity_dict[relation] = entities_name
                else:
                    entities = self.find_enetity(eid, relation, False)
                    entities = [item.replace("http://rdf.freebase.com/ns/","") for item in entities]
                    entities = [entity for entity in entities if entity.startswith("m.")]
                    entities = list(set(self.limit_list_size(entities)))
                    entities_name = [self.id2entity_name_or_type_new(item) for item in entities]
                    candidate_entity_list[id] = entities
                    all_candidate_entities.extend(entities)
                    all_candidate_entities_name.extend(entities_name)
                    entity_name_id_dict = self.set_key_value(entities_name, entities, entity_name_id_dict)
                    rel_entity_dict[relation] = entities_name
            all_candidate_entities_name_group[entity_idx].extend(all_candidate_entities_name)
            all_candidate_entities_group_list[entity_idx].extend(all_candidate_entities)
            rel_entity_dict_list.append(rel_entity_dict)
        all_candidate_entities_name_group_union = [item for sublist in all_candidate_entities_name_group \
                                             for item in (sublist if isinstance(sublist, list) else [sublist])]
        all_candidate_entities_name_group_union = self.clean_list(all_candidate_entities_name_group_union)
        most_relevant_entities = self.get_most_relevant_entities(all_candidate_entities_name_group_union, 
                                                                   entity_name, question, args,
                                                                   most_relevant_relations_list, knowpath_str)
        most_relevant_entities_list = self.extract_list_from_string(most_relevant_entities)
        most_relevant_entities_index = {}
        for relevant_ent in most_relevant_entities_list:
            group_id, relation, relation_pos = self.find_string_in_dict_list(relevant_ent, rel_entity_dict_list)
            most_relevant_entities_index[relevant_ent] = [group_id, relation, relation_pos]
        for e_id, e_name in enumerate(most_relevant_entities_index):
            if e_name not in entity_name_id_dict:
                continue
            group_id = most_relevant_entities_index[e_name][0]
            if group_id is None:
                continue
            # Find relation names
            find_relation = most_relevant_entities_index[e_name][1]
            path_is_head = list(enetity_dict.values())[group_id]
            
            is_head = head_or_tail_list_group[group_id][most_relevant_entities_index[e_name][2]]
            if depth == 0:
                path_is_head = not is_head
            # Update path
            path = self.update_path(path, path_is_head, is_head, group_id, find_relation, e_name, entity_name_id_dict)

        ops = NestedListOperations()
        extra_path  = ops.subtract(path, original_path, maintain_order=True)
        new_enetity_dict = {}
        new_entity_name = []
        for e_name in most_relevant_entities_index:
            if e_name not in entity_name_id_dict:
                    continue
            e_id_key = entity_name_id_dict[e_name]
            group_id = most_relevant_entities_index[e_name][0]
            true_or_false = list(enetity_dict.values())[group_id]
            if depth==0:
                is_head = head_or_tail_list_group[group_id][most_relevant_entities_index[e_name][2]]
                true_or_false = not is_head
            new_enetity_dict[e_id_key] = true_or_false
            new_entity_name.append(e_name)
        print(f"\nUpdated path: {extra_path}")
        print(f"\nUpdated entity list: {new_entity_name}")
        print(f"\nUpdated dictionary: {new_enetity_dict}")
        return extra_path, new_entity_name, new_enetity_dict

    def set_key_value(self, entities_name, entities, entity_name_id_dict):
        if len(entities) ==0 or len(entities_name) ==0:
            pass
        else:
            for id, item in enumerate(entities_name):
                if item != "UnName_Entity":
                    entity_name_id_dict[item] = entities[id].replace("http://rdf.freebase.com/ns/","")
        
        return entity_name_id_dict

    def update_path(self, path, path_is_head, is_head, group_id, find_relation, e_name, entity_name_id_dict):
        path_need_to_modify = path[group_id]
        path_bak = copy.deepcopy(path_need_to_modify)
        if not path_is_head:
            if not is_head:
                new_path = ['<-'] + [find_relation] + ['<-'] + [e_name]
                new_path = path_bak + new_path
            else:
                new_path = ['->'] + [find_relation] + ['->'] + [e_name]
                new_path = path_bak + new_path
        else:
            if not is_head:
                new_path = [e_name] + ['->'] + [find_relation] + ['->']
                new_path = new_path +path_bak
            else:
                new_path = [e_name] + ['<-'] + [find_relation] + ['<-']
                new_path = new_path +path_bak
        path.append(new_path)
        return path

    def clean_list(self, lst):
        return list(set([x for x in lst if x != 'UnName_Entity']))
    

    def find_elements_location(self, A, B):
        result = {}
        for item in A:
            for i, sublist in enumerate(B):
                if item in sublist:
                    result[item] = i
                    break
            else:
                result[item] = None  
        return result


    def find_string_in_dict_list(self, target_str, dict_list):
        for dict_idx, d in enumerate(dict_list):
            for key_idx, (key, value) in enumerate(d.items()):
                if isinstance(value, list) and target_str in value:
                    return dict_idx, key, key_idx
        return None, None, None

class NestedListOperations:
    def nested_list_subtract(self, list1, list2):
        def is_nested_equal(item1, item2):
            if isinstance(item1, list) and isinstance(item2, list):
                if len(item1) != len(item2):
                    return False
                return all(is_nested_equal(x, y) for x, y in zip(item1, item2))
            return item1 == item2

        result = []
        for item1 in list1:
            should_add = True
            for item2 in list2:
                if is_nested_equal(item1, item2):
                    should_add = False
                    break
            if should_add:
                result.append(item1)
        return result
    
    def subtract(self, list1, list2, maintain_order=True):
        def convert_to_tuple(lst):
            return tuple(convert_to_tuple(x) if isinstance(x, list) else x for x in lst)
        
        def convert_to_list(tup):
            return list(convert_to_list(x) if isinstance(x, tuple) else x for x in tup)
        
        if not maintain_order:
            set1 = {convert_to_tuple(x) if isinstance(x, list) else x for x in list1}
            set2 = {convert_to_tuple(x) if isinstance(x, list) else x for x in list2}
            result_set = set1 - set2
            return [convert_to_list(x) if isinstance(x, tuple) else x for x in result_set]
        else:
            return self.nested_list_subtract(list1, list2)
        


def half_stop(question, cluster_chain_of_entities, depth, args):
    """Abort search process midway

    Args:
        question (_type_): _description_
        cluster_chain_of_entities (_type_): _description_
        depth (_type_): _description_
        args (_type_): _description_
    """
    print("No new knowledge added during search depth %d, stop searching." % depth)
    answer = generate_answer(question, cluster_chain_of_entities, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, file_name=args.dataset, llm_type=args.LLM_type)


