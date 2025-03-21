
#' Evaluate data consistency by generating multiple LLM responses for one dataset
#'
#' @param prompts a list of prompts, generally made by subsampling the dataset on which you want to evaluate the consistancy
#' @param model A character string of the model name such as "llama3" with which you want to evaluate the prompts.
#'
#' @returns a list containing
#' * responses: a named list of all responses for all prompts
#' * central_answer: an answer meant to be the most representative one across all prompts
#' * pca: the result of a PCA computed on the embeddings of all responses
#' @export
#'
#' @examples
#' \dontrun{
#'
#'
##########################
# prompts comparison
##########################
#'
#'library(NaileR)
#'data(waste)
#'waste <- waste[-14]    # no variability on this question
#'
#'set.seed(1)
#'res_mca_waste <- MCA(waste, quali.sup = c(1,2,50:76),
#'                     ncp = 35, level.ventil = 0.05, graph = FALSE)
#'plot.MCA(res_mca_waste, choix = "ind",
#'         invisible = c("var", "quali.sup"), label = "none")
#'res_hcpc_waste <- HCPC(res_mca_waste, nb.clust = 3, graph = FALSE)
#'plot.HCPC(res_hcpc_waste, choice = "map", draw.tree = FALSE,
#'          ind.names = FALSE)
#'don_clust_waste <- res_hcpc_waste$data.clust
#'
#'intro_waste <- 'These data were collected
#'after a survey on food waste,
#'with participants describing their habits.'
#'intro_waste <- gsub('\n', ' ', intro_waste) |>
#'  stringr::str_squish()
#'
#'req_waste <- 'Please summarize the characteristics of each group.
#'Then, give each group a new name, based on your conclusions.
#'Finally, give each group a grade between 0 and 10,
#'based on how wasteful they are with food:
#'0 being "not at all", 10 being "absolutely".'
#'req_waste <- gsub('\n', ' ', req_waste) |>
#'  stringr::str_squish()
#'
#'res_waste <- nail_catdes(don_clust_waste,
#'                         num.var = ncol(don_clust_waste),
#'                         introduction = intro_waste,
#'                         request = req_waste,
#'                         quali.sample = 0.25,
#'                         quanti.sample = 1,
#'                         drop.negative = FALSE,
#'                         generate = FALSE,
#'                         proba = 0.25)
#'
#'cat(res_waste)
#'
#'prompt_list <- list()
#'num_repeats <- 5
#'
#'for (i in 1:num_repeats) {
#'  # Generate the prompt
#'
#'  prompt <- nail_catdes(don_clust_waste,
#'                        num.var = ncol(don_clust_waste),
#'                        introduction = intro_waste,
#'                        request = req_waste,
#'                        quali.sample = 0.35,
#'                        quanti.sample = 1,
#'                        drop.negative = FALSE,
#'                        generate = FALSE,
#'                        proba = 0.25)
#'
#'  # Store with a unique name
#'  prompt_list[[paste0("nail_catdes_", i)]] <- prompt
#'}
#'
#'res_prompt <- trustme_prompts(prompt_list, model = "llama2")
#'cat(res_prompt$central_answer)
#'
#' }
#'
#'
#'
#'
#'
#'
#'
#'
#'
trustme_prompts <- function(prompts, model) {

  results <- list()

  #if (!is.character(prompts) || length(prompts) == 0) {
  #  stop("Please provide a valid list of prompts.")
  #}

  if (missing(model) || !is.character(model)) {
    stop("Please specify a valid model.")
  }

  # Generate responses for each prompt
  for (name in names(prompts)) {
    response <- ollamar::generate(model = model, prompt = prompts[[name]], output = "text")
    results[[name]] <- response
  }

  # Convert responses into embeddings
  res_embeddings <- generate_embeddings(results)

  # Compute cosine distance matrix
  distance_matrix <- proxy::dist(as.data.frame(res_embeddings), method = "cosine")
  distance_matrix <- as.matrix(distance_matrix)

  # Identify the medoid (most central response)
  medoid_index <- cluster::pam(distance_matrix, k = 1)$medoids
  zero_indices <- which(medoid_index == 0)
  central_answer <- results[[zero_indices]]  # Extract central answer

  # PCA Analysis
  dta <- as.data.frame(res_embeddings)
  res.pca <- FactoMineR::PCA(dta, graph = FALSE)  # Ensure PCA is stored

  # Ensure PCA is displayed
  print(FactoMineR::plot.PCA(res.pca))

  # Return both results and the PCA object
  return(list(
    responses = results,
    central_answer = central_answer,
    pca = res.pca  # Return PCA object
  ))
}

