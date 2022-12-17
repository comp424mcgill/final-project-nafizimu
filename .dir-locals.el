((python-mode . ((eval . (dap-register-debug-template "Student Agent Debugger"
					       (list :type "python"
						     :args '("--player_1" "random_agent" "--player_2" "student_agent" "--display")
						     :cwd (projectile-project-root)
						     :env '(("VERBOSE" . "1"))
						     :target-module "simulator.py"
						     :request "launch"
						     :name "Student Agent Debugger"))))))
