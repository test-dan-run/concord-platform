import sys

if len(sys.argv) > 1 and sys.argv[1] == "import":
    from .importer import main as import_main

    sys.exit(import_main(sys.argv[2:]))

from .app import main

main()
