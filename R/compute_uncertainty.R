

# cosine_similarity <- function(sentence1, sentence2){
# 
# 
# 
#   # Otherwise, load and cache it
#   sentence_transformers <- reticulate::import("sentence_transformers")
#   model_sentence_transformer <- sentence_transformers$SentenceTransformer("all-mpnet-base-v2")
# 
# 
#   embedding1 <- model_sentence_transformer$encode(sentence1)
#   embedding2 <- model_sentence_transformer$encode(sentence2)
# 
#   output = sum(embedding1*embedding2)/(sqrt(sum(embedding1^2))*sqrt(sum(embedding2^2)))
# 
#   return(output)
# }




#' Measure uncertainty of LLM responses
#'
#' @param vec_text A vector of text for which the global uncertainty has to be measured
#' @param embedding_model A character string giving the Hugging Face identifier of the sentence-embedding model to use (e.g. "intfloat/e5-large-v2"). The model is loaded via SentenceTransformers and downloaded automatically if not already cached locally.
#'
#'
#' @returns a list containing
#' * laplacian_normalized: the normalized Laplacian of the similarity matrix
#' * similarity_matrix: the similarity matrix
#' * W: the symetrized similarity matrix
#' * UeigV: The "number of semantic meanings"
#' * eigen_values: the eigen values of the Laplacian
#' * Udeg: The average pairwise distance between responses
#' * Cdeg: for each node, the confidence score
#' * Uecc: average eccentricity
#' * Cecc: eccentricity for each node (useful for comparison between nodes)


#' @details
#' This function computes metrics based on the article of Lin et al., 2024. It embeds responses using the
#' [intfloat/e5-large-v2](https://huggingface.co/intfloat/multilingual-e5-large) model (by default, but can be modified), and then compute cosine similarities between them.
#' Several metrics are derived from the adjacency matrix built from the cosine similarities. Some are designed to quantify the global uncertainty (UeigV, Udeg, Uecc)
#' and other are designed to quantify and compare the uncertainty of each node (Cdeg, Cecc).
#'
#' @references
#'  Lin, Z., Trivedi, S., & Sun, J. (2024). Generating with confidence: Uncertainty quantification for black-box large language models
#'
#' @export
#'
#'
#' @examples
#' \dontrun{
#' # Processing time is often longer than ten seconds
#' # because the function uses multiple times a large language model.
#' library(NaileR)
#' data(beard_cont)
#' intro_beard <- 'A survey was conducted about beards
#' and 8 types of beards were described.
#' In the data that follow, beards are named B1 to B8.'
#' intro_beard <- gsub('\n', ' ', intro_beard) |>
#'   stringr::str_squish()
#'
#' req_beard <- 'Please give a name to each beard
#' and summarize what makes this beard unique.'
#' req_beard <- gsub('\n', ' ', req_beard) |>
#'   stringr::str_squish()
#'
#' res_trustme <- trustme_models(res_beard, models = c("llama2", "llama3", "llama3.1","mistral"))
#'
#' uncertainty <- compute_uncertainty_metrics(unlist(res_trustme$responses))
#'
#'
#'}
#'
#'
#'

compute_uncertainty_metrics <- function(vec_text, embedding_model = "intfloat/e5-large-v2"){

  
  
  if (!reticulate::py_module_available("sentence_transformers")) {
    stop("Le module Python 'sentence-transformers' is not installed. 
       Please install it using reticulate::py_install().")
  }
  
  
  if (!reticulate::py_module_available("transformers")) {
    stop("Le module Python 'transformers' is not installed. 
       Please install it using reticulate::py_install().")
  }
  
  
  
    
  transformers <- reticulate::import("transformers", convert = FALSE)
  sentence_transformers <- reticulate::import("sentence_transformers")
  model_sentence_transformer <- sentence_transformers$SentenceTransformer(embedding_model)
  
  
  
  
  
  embeddings <- model_sentence_transformer$encode(vec_text)
  embeddings <- embeddings / sqrt(rowSums(embeddings^2))
  
  # cosine sim = scalar prod
  similarity_matrix <- embeddings %*% t(embeddings)
  
  
  m = length(vec_text)
  
  if(is.null(names(vec_text))) {names(vec_text) = paste0("text", 1:m)}
  
  
  colnames(similarity_matrix) = rownames(similarity_matrix) = names(vec_text)
  
  
  # eq 5-6-7 paper Lin2024
  #adj matrix
  # useless if similarity matrix stays such
  W = (similarity_matrix + t(similarity_matrix)) / 2
  
  #degree_matrix
  D = diag(apply(W, 1, function(x) sum(x) ))
  
  
  #equation 8
  my_trace= function(mat_X) sum(diag(mat_X))
  
  Udeg = my_trace(m*diag(m) - D)/m^2
  Cdeg = diag(D)/m
  
  # equation 5
  laplacian_normalized = diag(ncol(W)) - MASS::ginv(sqrt(D)) %*% W  %*% MASS::ginv(sqrt(D))
  
  eigen_values = sort(eigen(laplacian_normalized)$
                        values)
  
  
  
  # Compute eccentricity value (and calibrate)
  # equation 9
  
  eigen_vectors = eigen(laplacian_normalized)$vectors
  
  #center eigen vec
  offset = sweep(eigen_vectors, 2, rowMeans(eigen_vectors))
  
  Cecc = apply(offset, 2, function(x) -sqrt(sum(x^2)))
  
  Uecc = sqrt(sum(offset^2))
  
  return(
    list(laplacian_normalized = laplacian_normalized,
         similarity_matrix = similarity_matrix,
         UeigV = sum(pmax(0,1 - eigen(laplacian_normalized)$values )  ),
         eigen_values = eigen(laplacian_normalized)$values,
         Udeg = Udeg,
         Cdeg = Cdeg,
         Uecc = Uecc,
         Cecc  = Cecc,
         W = W))
  
  
  
}



#' Plot uncertainty of LLM responses
#'
#' @param res_uncertainty The result of compute_ncertainty function
#'
#' @returns a list containing
#' * plot_graph: a plot of the graph with similarity between responses as adjacency matrix
#' * plot_eccentricity: a barplot displaying the eccentrictiy of each response
#' * plot_degree: a barplot displaying the degree (\eqn{\approx} confidence) in each response
#' * plot_uni_metrics: a barplot displaying different univariate metrics
#'
#'
#' @export
#'
#'
#' @examples
#' \dontrun{
#' # Processing time is often longer than ten seconds
#' # because the function uses multiple times a large language model.
#' library(NaileR)
#' data(beard_cont)
#' intro_beard <- 'A survey was conducted about beards
#' and 8 types of beards were described.
#' In the data that follow, beards are named B1 to B8.'
#' intro_beard <- gsub('\n', ' ', intro_beard) |>
#'   stringr::str_squish()
#'
#' req_beard <- 'Please give a name to each beard
#' and summarize what makes this beard unique.'
#' req_beard <- gsub('\n', ' ', req_beard) |>
#'   stringr::str_squish()
#'
#' res_trustme <- trustme_models(res_beard, models = c("llama2", "llama3", "llama3.1","mistral"))
#'
#' uncertainty <- compute_uncertainty_metrics(unlist(res_trustme$responses))
#' plot_uq <- plot_uncertainty(uncertainty)
#'
#'
#'}
#'

plot_uncertainty <- function(res_uncertainty){


  name_responses = colnames(res_uncertainty$W)
  graph <- igraph::graph_from_adjacency_matrix(res_uncertainty$W, mode = "undirected", weighted = TRUE)
  graph_tbl <- tidygraph::as_tbl_graph(graph)

  plot_graph = ggraph::ggraph(graph_tbl, layout = "circle") +
    # plot_graph = ggraph(graph_tbl, layout = "linear") +
    ggraph::geom_edge_link(ggplot2::aes(label = round(weight, 2)),
                   angle_calc = "along", label_dodge = ggplot2::unit(2.5, 'mm'),
                   label_size = 3, edge_colour = "gray50") +
    ggraph::geom_node_point(size = 5, color = "steelblue") +
    ggraph::geom_node_text(ggplot2::aes(label = name), vjust = -1) +
    ggplot2::theme_void()+
    ggplot2::labs(title = "Connections between responses")




  plot_eccentricity = tibble::tibble(response = name_responses,
                             eccentricity = c(res_uncertainty$Cecc)) %>%
    dplyr::mutate(response = forcats::fct_reorder(response, eccentricity )) %>%
    ggplot2::ggplot(ggplot2::aes(x = response, y = eccentricity)) +
    ggplot2::geom_col(position = "dodge", fill = "sienna", alpha =.4)+
    ggplot2::labs(title = "Eccentricity coefficient of each response")

  plot_degree = tibble::tibble(response = name_responses,
                       degree = c(res_uncertainty$Cdeg)) %>%
    dplyr::mutate(response = forcats::fct_reorder(response, degree )) %>%
    ggplot2::ggplot(ggplot2::aes(x = response, y = degree)) +
    ggplot2::geom_col(position = "dodge", fill = "forestgreen", alpha =.4) +
    ggplot2::labs(title = "Degree of each response")


  plot_uni_metrics = tibble::tibble(metric = c("UeigV","Udeg", "Uecc"),
                            value = c(res_uncertainty$UeigV, res_uncertainty$Udeg, res_uncertainty$Uecc))%>%
    dplyr::mutate(metric = forcats::fct_reorder(metric, value, .desc = TRUE)) %>%
    ggplot2::ggplot(ggplot2::aes(x = metric, y = value, fill = metric )) +
    ggplot2::geom_col(position = "dodge", alpha = .4) +
    ggplot2::scale_fill_manual(values = c("Udeg" = "forestgreen", "Uecc" = "sienna", "UeigV" = "steelblue3")) +
    ggplot2::labs(title ="Univariate metrics") +
    ggplot2::theme(legend.position = "none")

  return(list(plot_graph = plot_graph,
              plot_eccentricity = plot_eccentricity,
              plot_degree = plot_degree,
              plot_uni_metrics= plot_uni_metrics))



}

