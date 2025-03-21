#' Evaluate uncertainty of LLM responses (one model and multiple repetitions, or multiple models and on repetition)
#'
#' @param prompt a prompt to be given to the LLM
#' @param models A vector of characters strings containing names of different LLM that have to be tested. If length(models) > 1, then num_repeats has to be equal to 1 (either multiple models with one repetition each or one model with multiple repetitions.)
#'
#' @param num_repeats an integer specifying the number of times you want to evaluate the same prompt. If num_repeats > 1, then length(models) has to be equal to 1 (either multiple models with one repetition each or one model with multiple repetitions.)
#'
#' @returns a list containing
#' * responses: a named list of all responses of all models or repetitions
#' * central_answer: an answer meant to be the most representative one across all models or repetitions
#' * pca: the result of a PCA computed on the embeddings of all responses
#' * plot_pca: the two first principal components of the PCA
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
#' names(res_trustme)
#' res_trustme$central_answer
#'
#'
#'}
#'
trustme_models <- function(prompt, models = NULL, num_repeats = 1) {

  results <- list()

  if (is.null(models) || length(models) == 0) {
    stop("Please specify at least one model.")
  }

  if (length(models) > 1 && num_repeats > 1) {
    stop("Choose either multiple models with one repetition each or one model with multiple repetitions.")
  }

  if (length(models) == 1) {
    # One model, multiple repetitions
    model <- models[1]
    for (i in 1:num_repeats) {
      response <- ollamar::generate(model = model, prompt = prompt, output = "text")
      results[[paste0(model, "_", i)]] <- response
    }
  } else {
    # Multiple models, one repetition each
    for (model in models) {
      response <- ollamar::generate(model = model, prompt = prompt, output = "text")
      results[[model]] <- response
    }
  }

  res_embeddings <- generate_embeddings(results)
  distance_matrix <- proxy::dist(as.data.frame(res_embeddings), method = "cosine")
  distance_matrix <- as.matrix(distance_matrix)
  medoid_index <- cluster::pam(distance_matrix, k = 1)$medoids
  zero_indices <- which(medoid_index == 0)

  central_answer <- results[zero_indices]

  dta <- as.data.frame(res_embeddings)
  res.pca <- FactoMineR::PCA(dta, graph = F)
  pca_plot <- FactoMineR::plot.PCA(res.pca)

  return(list(
    responses = results,
    central_answer = central_answer,
    pca = res.pca,
    pca_plot = pca_plot
  ))
}



#' Generate embeddings from a list of texts
#'
#' @param text_list a list of texts
#' @param model Character string specifying pre-trained language model (default 'bert-base-uncased'). For full list of options see pretrained models at [HuggingFace](https://huggingface.co/). For example use "bert-base-multilingual-cased", "openai-gpt", "gpt2", "ctrl", "transfo-xl-wt103", "xlnet-base-cased", "xlm-mlm-enfr-1024", "distilbert-base-cased", "roberta-base", or "xlm-roberta-base". Only load models that you trust from HuggingFace; loading a malicious model can execute arbitrary code on your computer).
#'
#' @returns a dataframe containing the embedding of all texts submitted to the function
#'
generate_embeddings <- function(text_list, model = "bert-base-uncased") {

  # Ensure text_list is a list
  if (!is.list(text_list)) {
    stop("Input must be a list of text elements.")
  }

  # Function to embed a single text element
  embed_single_text <- function(text) {
    if (is.na(text) || text == "") {
      return(rep(NA, 768))  # Assuming 768 dimensions for BERT embeddings
    }
    embedding <- text::textEmbed(text, model = model)$texts
    return(as.numeric(unlist(embedding)))
  }

  # Generate embeddings for each list element
  embeddings_list <- lapply(text_list, embed_single_text)

  # Convert to dataframe
  embeddings_df <- as.data.frame(do.call(rbind, embeddings_list))

  # Assign column names starting with the model name
  colnames(embeddings_df) <- paste0(model, "_Dim", seq_len(ncol(embeddings_df)))

  return(embeddings_df)
}
