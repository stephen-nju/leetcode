add_requires("doctest")
add_rules("mode.release","mode.debug")
target("test")
    set_kind("binary")
    add_files("src/*.cc") 
    add_packages("doctest")

