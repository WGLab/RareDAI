from collections import Counter, defaultdict
from tqdm.auto import tqdm
from datetime import datetime
import json, re, ast
import pandas as pd
import numpy as np

class COTProcessing:
    def __init__(self):
        # self.text_dir = text_dir
        # self.text = text
        pass

    def read_text(self, text_dir = None, text = None):
        if text_dir is None and text is None:
            raise ImportError("No text file or text string is provided.")
        elif text is None:
            with open(text_dir, 'r') as f:
                text = f.readlines()
            self.result = text[0]
        else:
            self.result = text
        return self.result
    def first_pattern(self, string, pattern_A, pattern_B):
        # Find the first occurrence of pattern A
        match_A = re.search(pattern_A, string)
        
        # Find the first occurrence of pattern B
        match_B = re.search(pattern_B, string)
        
        if match_A is None and match_B is None:
            return np.nan
        elif match_A is None:
            return pattern_B
        elif match_B is None:
            return pattern_A
        else:
            # Compare the starting positions of both matches
            if match_A.start() < match_B.start():
                return pattern_A
            else:
                return pattern_B
    def get_answer_alternative(self, prediction):
        sentences = prediction.split("\n")
        answer_sen = [x for x in sentences if "I strongly recommend" in x or "I recommend" in x or ("most suitable" in x and "not" not in x)]
        if len(answer_sen) > 0:
            answer_sen = answer_sen[0]
            answer_sen = answer_sen.split(".")[0]
            self.label = first_pattern(answer_sen, 'gene panel', 'genome sequencing')
            return self.label
        else:
            self.label = ''
            return self.label
    def normalize_json_format(self, prediction, label = None):
        if "|==|Response|==|" in prediction:
            prediction = prediction.split("|==|Response|==|")[0]
        if "Response" in prediction:
            prediction = prediction.split("Response")[0]
        if label is None:
            label = self.label
        return {'cot':prediction, 'label':label}

## ICD10 PROCESSING
def ICDprocessing:
    def __init__(self):
        self.icd102phecode = self.convert_phecode()
    def convert_phecode():
        phecode_df = pd.read_csv('phecode_definitions1.2.csv', sep = ',')
        phecode_df = phecode_df[['phecode','phenotype']]
        phecode_icd = pd.read_csv('Phecode_map_v1_2_icd10_beta.csv',sep = ',')
        phecode_icd = phecode_icd[['ICD10','PHECODE']]
        merged_phecode = pd.merge(phecode_df, phecode_icd, left_on = 'phecode', right_on = 'PHECODE', how = 'inner')
        merged_phecode = merged_phecode[['ICD10', 'phenotype']]
        icd102phecode = merged_phecode.set_index('ICD10')['phenotype'].to_dict()
        return icd102phecode
    def generate_phecode(code):
        if code[:3] in self.icd102phecode.keys():
            phecode = self.icd102phecode[code[:3]]
            else:
                return np.nan
            if phecode.lower() == "other tests":
                return code
            else:
                return phecode
    def output_format(icd_list):
        if len(icd_list) > 0:
            return " | ".join(icd_list)
        else:
            return "Please refer the note for more detailed information."
# Load the HPO ontology
url = 'http://purl.obolibrary.org/obo/hp.obo'
graph = obonet.read_obo(url)
# Get direct children of Phenotypic Abnormality (HP:0000118)
direct_children = set(graph.successors("HP:0000118"))
second_level_children = set()
for child in direct_children:
    second_level_children.update(graph.successors(child))
class HPOprocessing:
    def __init__(self):
        pass
        
    # Function to get children of an HPO term
    def get_children(hpo_id):
        return nx.ancestors(graph, hpo_id)

    # Function to get immediate parents of an HPO term
    def get_parents(hpo_id):
        parent_nodes = list(nx.predecessor(graph, hpo_id).keys())
        parent_nodes.remove(hpo_id)
        return parent_nodes

    # Filter out redundant terms based on parent-child relationships
    def filter_redundant_terms(hpo_ids):
        filtered_terms = []
        
        for term in hpo_ids:
            # Skip if term is a direct child of HP:0000118
            if term in direct_children or term in second_level_children:
                continue
            
            is_redundant = False
            # if we want to keep detailed term, uncomment below:
    #         for selected in filtered_terms:
    #             try:
    #                 if term in get_children(selected):
    #                     is_redundant = False #meaning selected here is broader and need to be placed by the more detailed term
    #                     filtered_terms.remove(selected)
    #                 elif selected in get_children(term):
    #                     is_redundant = True # meaning the selected is already detailed and term is general, so no need to add
    #                     break
    #                 # if not fall into both cases above, then this term is new => add
    #             except:
    #                 pass # some terms are in old dictionary so just add
            # if we want to keep broader term (to reduce number of phenotypes), uncomment below:
            for selected in filtered_terms:
                try:
                    if term in get_children(selected):
                        is_redundant = True # we don't want to keep most detailed term
                        break
                    elif selected in get_children(term):
                        is_redundant = False # meaning the selected is already detailed and term is general, so we need to replace the selected with the term
                        filtered_terms.remove(selected)
                    # if not fall into both cases above, then this term is new => add
                except:
                    pass
            if not is_redundant:
                filtered_terms.append(term)
        
        return filtered_terms

    def filter_sibling_terms(hpo_ids, parent_node = 0):
        hierchary_dict = {}
        
        # Group terms by their grandparent terms
        for term in hpo_ids:
            try:
                # Get parent and grandparent of the term
                hierchary = get_parents(term)[parent_node] # 0 for parent, 1 for grandparent
            # Group by grandparent
                if hierchary not in hierchary_dict:
                    hierchary_dict[hierchary] = []
                hierchary_dict[hierchary].append(term)
            except:
                pass
        
        # Randomly select one term from each grandparent group
        reduced_terms = []
        for siblings in hierchary_dict.values():
            if len(siblings) > 1:
                # Randomly choose one sibling to keep
                selected_term = random.choice(siblings)
                reduced_terms.append(selected_term)
            else:
                # If there's only one term, keep it
                reduced_terms.extend(siblings)
        
        return reduced_terms
    # Step 2: Get a subgraph for the list of HPO IDs
    def get_hpo_subgraph(graph, hpo_ids):
        # Create a subgraph containing only the HPO IDs of interest
        subgraph = graph.subgraph(hpo_ids).copy()
        return subgraph

    # Step 3: Convert the directed graph to an undirected graph (if necessary)
    def convert_to_undirected(graph):
        # Convert the directed graph to an undirected graph
        return graph.to_undirected()

    # Step 4: Cluster the graph using connected components
    def cluster_graph(graph):
        # Using connected components to find clusters
        clusters = list(nx.connected_components(graph))
        return clusters

    # Main script
    def cluster(hpo_ids):
        # Step 1: Extract a subgraph for the list of HPO IDs
        hpo_subgraph = get_hpo_subgraph(graph, hpo_ids)

        # Step 2: Convert to undirected graph if it's directed
        if nx.is_directed(hpo_subgraph):
            hpo_subgraph = convert_to_undirected(hpo_subgraph)

        # Step 3: Cluster the graph using connected components
        clusters = cluster_graph(hpo_subgraph)
        
        # final clusters
        final_clusters = []
        for c in clusters:
            if len(c) > 1:
                c = list(c)
                pick_one_cluster = random.choice(c)
                final_clusters.append(pick_one_cluster)
            else:
                final_clusters.append(list(c)[0])
                
        # Output the clusters
        return final_clusters
    def reduce_phenotypes(hpo_ids):
        # Filter redundant terms based on parent-child relationships
        filtered_terms = filter_redundant_terms(hpo_ids)

        # Further filter out sibling terms
        removed_siblings = filter_sibling_terms(filtered_terms, parent_node = 0)
        removed_children = filter_sibling_terms(filtered_terms, parent_node = 1)
        #removed_siblings = filter_sibling_terms(filtered_terms, parent_node = 0)
        
        # cluster
        final_list = cluster(removed_children)
        return final_list
    def output_format(final_list):
        if len(final_list) > 0:
            return " | ".join(final_list)
        else:
            return "Please refer the note for more detailed information."