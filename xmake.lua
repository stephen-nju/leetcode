add_requires("doctest")
set_languages("c99","cxx14")
add_rules("mode.release","mode.debug")
target("test")
    set_kind("binary")
    set_symbols("debug")
    add_files("src/*.cc") 
    add_packages("doctest")

