
# TrustMe

<!-- badges: start -->
<!-- badges: end -->

The goal of TrustMe is to enhance confidence in LLM-generated responses, particularly within
the [NaileR](https://cran.r-project.org/web/packages/NaileR/index.html) ecosystem.

## Installation

You can install the development version of TrustMe like so:

``` r
# FILL THIS IN! HOW CAN PEOPLE INSTALL YOUR DEV PACKAGE?
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(TrustMe)
library(NaileR)

data(beard_cont)
intro_beard <- 'A survey was conducted about beards
and 8 types of beards were described.
In the data that follow, beards are named B1 to B8.'
intro_beard <- gsub('\n', ' ', intro_beard) |>
  stringr::str_squish()

req_beard <- 'Please give a name to each beard
and summarize what makes this beard unique.'
req_beard <- gsub('\n', ' ', req_beard) |>
  stringr::str_squish()

# prompt generation
res_beard <- nail_descfreq(beard_cont,
                           introduction = intro_beard,
                           request = req_beard,
                           generate = FALSE)
res_beard

#######################
# models comparison
#######################

res_trustme <- trustme_models(res_beard, models = c("llama2", "llama3", "llama3.1","mistral"))
names(res_trustme)
cat(res_trustme$central_answer[[1]])
res_trustme$pca_plot

##########################
# repetitions comparison
##########################
res_trustme <- trustme_models(res_beard, models = "llama2", num_repeats = 5)
cat(res_trustme$central_answer[[1]])

```

