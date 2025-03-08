#!/usr/bin/env python3
import argparse
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Import des composants du système
from rag_pipeline_orchestrator import RAGPipelineOrchestrator
from config import get_config, save_config, logger

def parse_arguments():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description="Application RAG pour la recherche et le dialogue avec les documents")
    
    # Commandes principales
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande: process
    process_parser = subparsers.add_parser("process", help="Traiter les documents")
    process_parser.add_argument("--dir", help="Répertoire à traiter (sous-répertoire de data_dir)")
    process_parser.add_argument("--force", action="store_true", help="Forcer le retraitement des documents")
    process_parser.add_argument("--async", dest="async_mode", action="store_true", help="Exécuter en mode asynchrone")
    
    # Commande: index
    index_parser = subparsers.add_parser("index", help="Indexer les documents")
    index_parser.add_argument("--force", action="store_true", help="Forcer la réindexation des documents")
    index_parser.add_argument("--async", dest="async_mode", action="store_true", help="Exécuter en mode asynchrone")
    
    # Commande: search
    search_parser = subparsers.add_parser("search", help="Rechercher dans les documents")
    search_parser.add_argument("query", help="Requête de recherche")
    search_parser.add_argument("--top", type=int, default=5, help="Nombre de résultats à retourner")
    search_parser.add_argument("--type", choices=["semantic", "hybrid"], default="hybrid", help="Type de recherche")
    search_parser.add_argument("--weight", type=float, help="Poids sémantique pour la recherche hybride (0-1)")
    search_parser.add_argument("--rerank", action="store_true", help="Activer le réordonnancement")
    search_parser.add_argument("--filter", action="append", help="Filtres au format clé:valeur")
    search_parser.add_argument("--all-chunks", action="store_true", help="Retourner tous les chunks individuels")
    
    # Commande: ask
    ask_parser = subparsers.add_parser("ask", help="Poser une question au système")
    ask_parser.add_argument("question", help="Question à poser")
    ask_parser.add_argument("--conversation", help="ID de la conversation")
    ask_parser.add_argument("--top", type=int, help="Nombre de résultats à utiliser")
    ask_parser.add_argument("--type", choices=["semantic", "hybrid"], help="Type de recherche")
    
    # Commande: conversation
    conv_parser = subparsers.add_parser("conversation", help="Gérer les conversations")
    conv_subparsers = conv_parser.add_subparsers(dest="conv_command", help="Commande de conversation")
    
    # Sous-commande: new
    conv_new_parser = conv_subparsers.add_parser("new", help="Créer une nouvelle conversation")
    
    # Sous-commande: list
    conv_list_parser = conv_subparsers.add_parser("list", help="Lister les conversations")
    
    # Sous-commande: show
    conv_show_parser = conv_subparsers.add_parser("show", help="Afficher une conversation")
    conv_show_parser.add_argument("id", help="ID de la conversation")
    
    # Sous-commande: delete
    conv_delete_parser = conv_subparsers.add_parser("delete", help="Supprimer une conversation")
    conv_delete_parser.add_argument("id", help="ID de la conversation")
    
    # Commande: stats
    stats_parser = subparsers.add_parser("stats", help="Afficher les statistiques du système")
    
    # Commande: config
    config_parser = subparsers.add_parser("config", help="Gérer la configuration")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Commande de configuration")
    
    # Sous-commande: show
    config_show_parser = config_subparsers.add_parser("show", help="Afficher la configuration")
    config_show_parser.add_argument("--section", help="Section spécifique à afficher")
    
    # Sous-commande: update
    config_update_parser = config_subparsers.add_parser("update", help="Mettre à jour la configuration")
    config_update_parser.add_argument("key", help="Clé à mettre à jour (format: section.sous_section.clé)")
    config_update_parser.add_argument("value", help="Nouvelle valeur")
    
    # Sous-commande: import
    config_import_parser = config_subparsers.add_parser("import", help="Importer une configuration depuis un fichier")
    config_import_parser.add_argument("file", help="Fichier de configuration JSON")
    
    return parser.parse_args()

def format_json_output(data: Any, pretty: bool = True) -> str:
    """Formate les données en JSON pour l'affichage"""
    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data)

def parse_filters(filter_strings: List[str]) -> Dict[str, Any]:
    """Parse les filtres au format clé:valeur"""
    if not filter_strings:
        return None
    
    filters = {}
    for filter_str in filter_strings:
        if ":" in filter_str:
            key, value = filter_str.split(":", 1)
            # Tenter de convertir les valeurs numériques
            try:
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Conserver la valeur telle quelle si ce n'est pas un nombre
                pass
            
            filters[key] = value
    
    return filters if filters else None

def handle_process_command(orchestrator: RAGPipelineOrchestrator, args):
    """Traite la commande de traitement des documents"""
    print(f"Traitement des documents{f' du répertoire {args.dir}' if args.dir else ''}...")
    
    result = orchestrator.process_documents(
        subdirectory=args.dir,
        force_reprocess=args.force,
        async_mode=args.async_mode
    )
    
    if args.async_mode:
        print(f"Tâche de traitement lancée: {result['task_id']}")
    else:
        print(f"Traitement terminé: {result['processed_documents']} documents traités, "
              f"{result['skipped_documents']} ignorés, "
              f"{result['failed_documents']} en échec")

def handle_index_command(orchestrator: RAGPipelineOrchestrator, args):
    """Traite la commande d'indexation des documents"""
    print("Indexation des documents...")
    
    result = orchestrator.index_documents(
        force_reindex=args.force,
        async_mode=args.async_mode
    )
    
    if args.async_mode:
        print(f"Tâche d'indexation lancée: {result['task_id']}")
    else:
        print(f"Indexation terminée: {result['indexed_documents']} documents indexés, "
              f"{result['skipped_documents']} ignorés, "
              f"{result['failed_documents']} en échec")
        print(f"Temps total: {result['total_processing_time']} secondes")

def handle_search_command(orchestrator: RAGPipelineOrchestrator, args):
    """Traite la commande de recherche"""
    print(f"Recherche: '{args.query}'")
    
    # Parser les filtres
    filters = parse_filters(args.filter)
    
    # Effectuer la recherche
    result = orchestrator.search(
        query=args.query,
        top_k=args.top,
        filters=filters,
        search_type=args.type,
        semantic_weight=args.weight,
        rerank_results=args.rerank,
        return_all_chunks=args.all_chunks
    )
    
    # Afficher les résultats
    print(f"Résultats ({result['total_results']} trouvés, {result['processing_time_seconds']}s):")
    
    if args.all_chunks:
        # Afficher tous les chunks individuels
        for i, chunk in enumerate(result["results"], 1):
            print(f"\n{i}. Chunk: {chunk['chunk_id']} (score: {chunk['combined_score']:.4f})")
            print(f"   Document: {chunk['document_id']}")
            print(f"   Texte: {chunk['text'][:200]}...")
    else:
        # Afficher les résultats groupés par document
        for i, doc in enumerate(result["results"], 1):
            print(f"\n{i}. Document: {doc['document_id']} (score max: {doc['max_score']:.4f})")
            if "metadata" in doc and "filename" in doc["metadata"]:
                print(f"   Fichier: {doc['metadata']['filename']}")
            print(f"   Chunks: {len(doc['chunks'])}")
            if doc['chunks']:
                print(f"   Extrait: {doc['chunks'][0]['text'][:200]}...")
    
    print(f"\nType de recherche: {args.type}" + 
          (f", Poids sémantique: {result['semantic_weight']}" if args.type == "hybrid" else "") +
          (f", Réordonnement: {'Oui' if result['reranked'] else 'Non'}"))

def handle_ask_command(orchestrator: RAGPipelineOrchestrator, args):
    """Traite la commande de question"""
    print(f"Question: {args.question}")
    
    # Paramètres de recherche personnalisés
    search_params = {}
    if args.top:
        search_params["top_k"] = args.top
    if args.type:
        search_params["search_type"] = args.type
    
    # Poser la question
    response = orchestrator.ask(
        question=args.question,
        conversation_id=args.conversation,
        search_params=search_params
    )
    
    # Afficher la réponse
    print(f"\nConversation: {response['conversation_id']}")
    print(f"\nRéponse:\n{response['answer']}\n")
    
    # Afficher les sources
    if "sources" in response and response["sources"]:
        print("Sources:")
        for i, source in enumerate(response["sources"], 1):
            print(f"  {i}. Document: {source['document_id']}, Chunk: {source['chunk_id']}")
            if "page" in source and source["page"]:
                print(f"     Page: {source['page']}")
            print(f"     Aperçu: {source['text_preview']}")
    
    print(f"\nTemps de traitement: {response['processing_time_seconds']} secondes")

def handle_conversation_command(orchestrator: RAGPipelineOrchestrator, args):
    """Traite les commandes de gestion des conversations"""
    document_chat = orchestrator.document_chat
    
    if args.conv_command == "new":
        # Créer une nouvelle conversation
        conv_id = document_chat.new_conversation()
        print(f"Nouvelle conversation créée: {conv_id}")
    
    elif args.conv_command == "list":
        # Lister les conversations
        conversations = document_chat.list_conversations()
        
        if not conversations:
            print("Aucune conversation trouvée.")
            return
        
        print(f"Conversations ({len(conversations)}):")
        for i, conv in enumerate(conversations, 1):
            print(f"{i}. ID: {conv['id']}")
            print(f"   Créée le: {conv['created_at']}")
            print(f"   Dernière mise à jour: {conv['last_update']}")
            print(f"   Messages: {conv['message_count']}")
            if conv['preview']:
                print(f"   Aperçu: {conv['preview']}")
            print()
    
    elif args.conv_command == "show":
        # Afficher une conversation
        try:
            conversation = document_chat.get_conversation(args.id)
            
            print(f"Conversation: {conversation['id']}")
            print(f"Créée le: {conversation['created_at']}")
            print(f"Dernière mise à jour: {conversation['last_update']}")
            print(f"Messages: {len(conversation['messages'])}\n")
            
            for i, msg in enumerate(conversation['messages'], 1):
                print(f"{i}. [{msg['role']}] {msg['timestamp']}")
                print(f"   {msg['content']}")
                print()
        
        except ValueError as e:
            print(f"Erreur: {str(e)}")
    
    elif args.conv_command == "delete":
        # Supprimer une conversation
        success = document_chat.delete_conversation(args.id)
        if success:
            print(f"Conversation {args.id} supprimée avec succès.")
        else:
            print(f"Impossible de supprimer la conversation {args.id} (non trouvée).")

def handle_stats_command(orchestrator: RAGPipelineOrchestrator, args):
    """Affiche les statistiques du système"""
    state = orchestrator.get_state()
    
    print("État du pipeline RAG:")
    print(f"Status: {state['status']}")
    print(f"Dernière mise à jour: {state['last_update']}")
    
    print("\nComposants:")
    for comp_name, comp_state in state["components"].items():
        print(f"  {comp_name}: {comp_state['status']}")
    
    print("\nStatistiques:")
    stats = state["statistics"]
    print(f"  Documents: {stats['total_documents']}")
    print(f"  Chunks: {stats['total_chunks']}")
    print(f"  Requêtes: {stats['total_queries']}")
    
    if stats['last_processing_time']:
        print(f"  Dernier traitement: {stats['last_processing_time']}")
    
    if stats['last_indexing_time']:
        print(f"  Dernière indexation: {stats['last_indexing_time']}")
    
    # Statistiques détaillées de l'indexeur si disponible
    if orchestrator._vector_indexer:
        print("\nDétails de l'indexation:")
        idx_stats = orchestrator.vector_indexer.get_stats()
        print(f"  Base vectorielle: {idx_stats['vector_db_type']}")
        print(f"  Modèle d'embedding: {idx_stats['embedding_model']}")
        print(f"  Dispositif: {idx_stats['device']}")

def handle_config_command(orchestrator: RAGPipelineOrchestrator, args):
    """Gère les commandes de configuration"""
    if args.config_command == "show":
        # Afficher la configuration
        config = orchestrator.config
        
        if args.section:
            if "." in args.section:
                # Traverser la hiérarchie de la configuration
                path = args.section.split(".")
                current = config
                for key in path:
                    if key in current:
                        current = current[key]
                    else:
                        print(f"Section {args.section} non trouvée dans la configuration.")
                        return
                print(format_json_output(current))
            elif args.section in config:
                # Afficher une section spécifique
                print(format_json_output(config[args.section]))
            else:
                print(f"Section {args.section} non trouvée dans la configuration.")
        else:
            # Afficher toute la configuration
            print(format_json_output(config))
    
    elif args.config_command == "update":
        # Mettre à jour une valeur de configuration
        key_path = args.key.split(".")
        value = args.value
        
        # Tenter de convertir la valeur dans le type approprié
        try:
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif all(c.isdigit() or c == "." for c in value) and value.count(".") == 1:
                value = float(value)
        except (ValueError, AttributeError):
            pass
        
        # Construire la mise à jour de configuration
        update = {}
        current = update
        for i, key in enumerate(key_path):
            if i == len(key_path) - 1:
                current[key] = value
            else:
                current[key] = {}
                current = current[key]
        
        # Appliquer la mise à jour
        updated_config = orchestrator.configure(update)
        print(f"Configuration mise à jour: {args.key} = {value}")
    
    elif args.config_command == "import":
        # Importer une configuration depuis un fichier
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Erreur: Le fichier {file_path} n'existe pas.")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_updates = json.load(f)
            
            # Appliquer les mises à jour
            updated_config = orchestrator.configure(config_updates)
            print(f"Configuration importée depuis {file_path}")
        except Exception as e:
            print(f"Erreur lors de l'importation de la configuration: {str(e)}")

def main():
    """Fonction principale"""
    args = parse_arguments()
    
    # Initialiser l'orchestrateur du pipeline RAG
    orchestrator = RAGPipelineOrchestrator()
    
    # Traiter la commande
    if args.command == "process":
        handle_process_command(orchestrator, args)
    
    elif args.command == "index":
        handle_index_command(orchestrator, args)
    
    elif args.command == "search":
        handle_search_command(orchestrator, args)
    
    elif args.command == "ask":
        handle_ask_command(orchestrator, args)
    
    elif args.command == "conversation":
        handle_conversation_command(orchestrator, args)
    
    elif args.command == "stats":
        handle_stats_command(orchestrator, args)
    
    elif args.command == "config":
        handle_config_command(orchestrator, args)
    
    else:
        print("Commande non spécifiée. Utilisez --help pour afficher les options disponibles.")
    
    # Nettoyer les ressources
    orchestrator.cleanup()

if __name__ == "__main__":
    main()