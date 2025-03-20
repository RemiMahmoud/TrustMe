
usethis::use_package_doc()
usethis::use_roxygen_md()


list_packages = c("FactoMineR", "cluster", "ollamar", "text", "NaileR", "proxy", "stringr")
invisible(lapply(list_packages, usethis::use_package))



usethis::use_r("trustme_models")
usethis::use_roxygen_md()
