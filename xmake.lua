add_rules("mode.release","mode.debug")
target("main")
    set_kind("binary")
    add_files("src/*.cc") 

