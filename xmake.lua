add_requires("doctest")
add_rules("mode.release","mode.debug")
target("test")
    set_kind("binary")
    set_languages("C99","cxx11")
    add_files("src/*.cc") 
    add_packages("doctest")

