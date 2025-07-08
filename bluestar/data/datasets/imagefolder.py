import os
import shutil

class_list = [
  {
    "id": "n01440764",
    "superclass": "fish",
    "class": "tench"
  },
  {
    "id": "n01443537",
    "superclass": "fish",
    "class": "goldfish"
  },
  {
    "id": "n01484850",
    "superclass": "shark",
    "class": "great white shark"
  },
  {
    "id": "n01491361",
    "superclass": "shark",
    "class": "tiger shark"
  },
  {
    "id": "n01494475",
    "superclass": "shark",
    "class": "hammerhead"
  },
  {
    "id": "n01496331",
    "superclass": "fish",
    "class": "electric ray"
  },
  {
    "id": "n01498041",
    "superclass": "fish",
    "class": "stingray"
  },
  {
    "id": "n01514668",
    "superclass": "bird",
    "class": "cock"
  },
  {
    "id": "n01514859",
    "superclass": "bird",
    "class": "hen"
  },
  {
    "id": "n01518878",
    "superclass": "bird",
    "class": "ostrich"
  },
  {
    "id": "n01530575",
    "superclass": "bird",
    "class": "brambling"
  },
  {
    "id": "n01531178",
    "superclass": "bird",
    "class": "goldfinch"
  },
  {
    "id": "n01532829",
    "superclass": "bird",
    "class": "house finch"
  },
  {
    "id": "n01534433",
    "superclass": "bird",
    "class": "junco"
  },
  {
    "id": "n01537544",
    "superclass": "bird",
    "class": "indigo bunting"
  },
  {
    "id": "n01558993",
    "superclass": "bird",
    "class": "robin"
  },
  {
    "id": "n01560419",
    "superclass": "bird",
    "class": "bulbul"
  },
  {
    "id": "n01580077",
    "superclass": "bird",
    "class": "jay"
  },
  {
    "id": "n01582220",
    "superclass": "bird",
    "class": "magpie"
  },
  {
    "id": "n01592084",
    "superclass": "bird",
    "class": "chickadee"
  },
  {
    "id": "n01601694",
    "superclass": "bird",
    "class": "water ouzel"
  },
  {
    "id": "n01608432",
    "superclass": "bird",
    "class": "kite"
  },
  {
    "id": "n01614925",
    "superclass": "bird",
    "class": "bald eagle"
  },
  {
    "id": "n01616318",
    "superclass": "bird",
    "class": "vulture"
  },
  {
    "id": "n01622779",
    "superclass": "bird",
    "class": "great grey owl"
  },
  {
    "id": "n01629819",
    "superclass": "salamander",
    "class": "European fire salamander"
  },
  {
    "id": "n01630670",
    "superclass": "salamander",
    "class": "common newt"
  },
  {
    "id": "n01631663",
    "superclass": "salamander",
    "class": "eft"
  },
  {
    "id": "n01632458",
    "superclass": "salamander",
    "class": "spotted salamander"
  },
  {
    "id": "n01632777",
    "superclass": "salamander",
    "class": "axolotl"
  },
  {
    "id": "n01641577",
    "superclass": "frog",
    "class": "bullfrog"
  },
  {
    "id": "n01644373",
    "superclass": "frog",
    "class": "tree frog"
  },
  {
    "id": "n01644900",
    "superclass": "frog",
    "class": "tailed frog"
  },
  {
    "id": "n01664065",
    "superclass": "turtle",
    "class": "loggerhead"
  },
  {
    "id": "n01665541",
    "superclass": "turtle",
    "class": "leatherback turtle"
  },
  {
    "id": "n01667114",
    "superclass": "turtle",
    "class": "mud turtle"
  },
  {
    "id": "n01667778",
    "superclass": "turtle",
    "class": "terrapin"
  },
  {
    "id": "n01669191",
    "superclass": "turtle",
    "class": "box turtle"
  },
  {
    "id": "n01675722",
    "superclass": "lizard",
    "class": "banded gecko"
  },
  {
    "id": "n01677366",
    "superclass": "lizard",
    "class": "common iguana"
  },
  {
    "id": "n01682714",
    "superclass": "lizard",
    "class": "American chameleon"
  },
  {
    "id": "n01685808",
    "superclass": "lizard",
    "class": "whiptail"
  },
  {
    "id": "n01687978",
    "superclass": "lizard",
    "class": "agama"
  },
  {
    "id": "n01688243",
    "superclass": "lizard",
    "class": "frilled lizard"
  },
  {
    "id": "n01689811",
    "superclass": "lizard",
    "class": "alligator lizard"
  },
  {
    "id": "n01692333",
    "superclass": "lizard",
    "class": "Gila monster"
  },
  {
    "id": "n01693334",
    "superclass": "lizard",
    "class": "green lizard"
  },
  {
    "id": "n01694178",
    "superclass": "lizard",
    "class": "African chameleon"
  },
  {
    "id": "n01695060",
    "superclass": "lizard",
    "class": "Komodo dragon"
  },
  {
    "id": "n01697457",
    "superclass": "crocodile",
    "class": "African crocodile"
  },
  {
    "id": "n01698640",
    "superclass": "crocodile",
    "class": "American alligator"
  },
  {
    "id": "n01704323",
    "superclass": "dinosaur",
    "class": "triceratops"
  },
  {
    "id": "n01728572",
    "superclass": "snake",
    "class": "thunder snake"
  },
  {
    "id": "n01728920",
    "superclass": "snake",
    "class": "ringneck snake"
  },
  {
    "id": "n01729322",
    "superclass": "snake",
    "class": "hognose snake"
  },
  {
    "id": "n01729977",
    "superclass": "snake",
    "class": "green snake"
  },
  {
    "id": "n01734418",
    "superclass": "snake",
    "class": "king snake"
  },
  {
    "id": "n01735189",
    "superclass": "snake",
    "class": "garter snake"
  },
  {
    "id": "n01737021",
    "superclass": "snake",
    "class": "water snake"
  },
  {
    "id": "n01739381",
    "superclass": "snake",
    "class": "vine snake"
  },
  {
    "id": "n01740131",
    "superclass": "snake",
    "class": "night snake"
  },
  {
    "id": "n01742172",
    "superclass": "snake",
    "class": "boa constrictor"
  },
  {
    "id": "n01744401",
    "superclass": "snake",
    "class": "rock python"
  },
  {
    "id": "n01748264",
    "superclass": "snake",
    "class": "Indian cobra"
  },
  {
    "id": "n01749939",
    "superclass": "snake",
    "class": "green mamba"
  },
  {
    "id": "n01751748",
    "superclass": "fish",
    "class": "sea snake"
  },
  {
    "id": "n01753488",
    "superclass": "snake",
    "class": "horned viper"
  },
  {
    "id": "n01755581",
    "superclass": "snake",
    "class": "diamondback"
  },
  {
    "id": "n01756291",
    "superclass": "snake",
    "class": "sidewinder"
  },
  {
    "id": "n01768244",
    "superclass": "trilobite",
    "class": "trilobite"
  },
  {
    "id": "n01770081",
    "superclass": "arachnid",
    "class": "harvestman"
  },
  {
    "id": "n01770393",
    "superclass": "arachnid",
    "class": "scorpion"
  },
  {
    "id": "n01773157",
    "superclass": "arachnid",
    "class": "black and gold garden spider"
  },
  {
    "id": "n01773549",
    "superclass": "arachnid",
    "class": "barn spider"
  },
  {
    "id": "n01773797",
    "superclass": "arachnid",
    "class": "garden spider"
  },
  {
    "id": "n01774384",
    "superclass": "arachnid",
    "class": "black widow"
  },
  {
    "id": "n01774750",
    "superclass": "arachnid",
    "class": "tarantula"
  },
  {
    "id": "n01775062",
    "superclass": "arachnid",
    "class": "wolf spider"
  },
  {
    "id": "n01776313",
    "superclass": "bug",
    "class": "tick"
  },
  {
    "id": "n01784675",
    "superclass": "bug",
    "class": "centipede"
  },
  {
    "id": "n01795545",
    "superclass": "bird",
    "class": "black grouse"
  },
  {
    "id": "n01796340",
    "superclass": "bird",
    "class": "ptarmigan"
  },
  {
    "id": "n01797886",
    "superclass": "bird",
    "class": "ruffed grouse"
  },
  {
    "id": "n01798484",
    "superclass": "bird",
    "class": "prairie chicken"
  },
  {
    "id": "n01806143",
    "superclass": "bird",
    "class": "peacock"
  },
  {
    "id": "n01806567",
    "superclass": "bird",
    "class": "quail"
  },
  {
    "id": "n01807496",
    "superclass": "bird",
    "class": "partridge"
  },
  {
    "id": "n01817953",
    "superclass": "bird",
    "class": "African grey"
  },
  {
    "id": "n01818515",
    "superclass": "bird",
    "class": "macaw"
  },
  {
    "id": "n01819313",
    "superclass": "bird",
    "class": "sulphur-crested cockatoo"
  },
  {
    "id": "n01820546",
    "superclass": "bird",
    "class": "lorikeet"
  },
  {
    "id": "n01824575",
    "superclass": "bird",
    "class": "coucal"
  },
  {
    "id": "n01828970",
    "superclass": "bird",
    "class": "bee eater"
  },
  {
    "id": "n01829413",
    "superclass": "bird",
    "class": "hornbill"
  },
  {
    "id": "n01833805",
    "superclass": "bird",
    "class": "hummingbird"
  },
  {
    "id": "n01843065",
    "superclass": "bird",
    "class": "jacamar"
  },
  {
    "id": "n01843383",
    "superclass": "bird",
    "class": "toucan"
  },
  {
    "id": "n01847000",
    "superclass": "bird",
    "class": "drake"
  },
  {
    "id": "n01855032",
    "superclass": "bird",
    "class": "red-breasted merganser"
  },
  {
    "id": "n01855672",
    "superclass": "bird",
    "class": "goose"
  },
  {
    "id": "n01860187",
    "superclass": "bird",
    "class": "black swan"
  },
  {
    "id": "n01871265",
    "superclass": "ungulate",
    "class": "tusker"
  },
  {
    "id": "n01872401",
    "superclass": "monotreme",
    "class": "echidna"
  },
  {
    "id": "n01873310",
    "superclass": "monotreme",
    "class": "platypus"
  },
  {
    "id": "n01877812",
    "superclass": "marsupial",
    "class": "wallaby"
  },
  {
    "id": "n01882714",
    "superclass": "marsupial",
    "class": "koala"
  },
  {
    "id": "n01883070",
    "superclass": "marsupial",
    "class": "wombat"
  },
  {
    "id": "n01910747",
    "superclass": "fish",
    "class": "jellyfish"
  },
  {
    "id": "n01914609",
    "superclass": "coral",
    "class": "sea anemone"
  },
  {
    "id": "n01917289",
    "superclass": "coral",
    "class": "brain coral"
  },
  {
    "id": "n01924916",
    "superclass": "platyhelminthes",
    "class": "flatworm"
  },
  {
    "id": "n01930112",
    "superclass": "bug",
    "class": "nematode"
  },
  {
    "id": "n01943899",
    "superclass": "mollusk",
    "class": "conch"
  },
  {
    "id": "n01944390",
    "superclass": "mollusk",
    "class": "snail"
  },
  {
    "id": "n01945685",
    "superclass": "mollusk",
    "class": "slug"
  },
  {
    "id": "n01950731",
    "superclass": "mollusk",
    "class": "sea slug"
  },
  {
    "id": "n01955084",
    "superclass": "mollusk",
    "class": "chiton"
  },
  {
    "id": "n01968897",
    "superclass": "mollusk",
    "class": "chambered nautilus"
  },
  {
    "id": "n01978287",
    "superclass": "crustacean",
    "class": "Dungeness crab"
  },
  {
    "id": "n01978455",
    "superclass": "crustacean",
    "class": "rock crab"
  },
  {
    "id": "n01980166",
    "superclass": "crustacean",
    "class": "fiddler crab"
  },
  {
    "id": "n01981276",
    "superclass": "crustacean",
    "class": "king crab"
  },
  {
    "id": "n01983481",
    "superclass": "crustacean",
    "class": "American lobster"
  },
  {
    "id": "n01984695",
    "superclass": "crustacean",
    "class": "spiny lobster"
  },
  {
    "id": "n01985128",
    "superclass": "crustacean",
    "class": "crayfish"
  },
  {
    "id": "n01986214",
    "superclass": "crustacean",
    "class": "hermit crab"
  },
  {
    "id": "n01990800",
    "superclass": "crustacean",
    "class": "isopod"
  },
  {
    "id": "n02002556",
    "superclass": "bird",
    "class": "white stork"
  },
  {
    "id": "n02002724",
    "superclass": "bird",
    "class": "black stork"
  },
  {
    "id": "n02006656",
    "superclass": "bird",
    "class": "spoonbill"
  },
  {
    "id": "n02007558",
    "superclass": "bird",
    "class": "flamingo"
  },
  {
    "id": "n02009229",
    "superclass": "bird",
    "class": "little blue heron"
  },
  {
    "id": "n02009912",
    "superclass": "bird",
    "class": "American egret"
  },
  {
    "id": "n02011460",
    "superclass": "bird",
    "class": "bittern"
  },
  {
    "id": "n02012849",
    "superclass": "bird",
    "class": "crane"
  },
  {
    "id": "n02013706",
    "superclass": "bird",
    "class": "limpkin"
  },
  {
    "id": "n02017213",
    "superclass": "bird",
    "class": "European gallinule"
  },
  {
    "id": "n02018207",
    "superclass": "bird",
    "class": "American coot"
  },
  {
    "id": "n02018795",
    "superclass": "bird",
    "class": "bustard"
  },
  {
    "id": "n02025239",
    "superclass": "bird",
    "class": "ruddy turnstone"
  },
  {
    "id": "n02027492",
    "superclass": "bird",
    "class": "red-backed sandpiper"
  },
  {
    "id": "n02028035",
    "superclass": "bird",
    "class": "redshank"
  },
  {
    "id": "n02033041",
    "superclass": "bird",
    "class": "dowitcher"
  },
  {
    "id": "n02037110",
    "superclass": "bird",
    "class": "oystercatcher"
  },
  {
    "id": "n02051845",
    "superclass": "bird",
    "class": "pelican"
  },
  {
    "id": "n02056570",
    "superclass": "bird",
    "class": "king penguin"
  },
  {
    "id": "n02058221",
    "superclass": "bird",
    "class": "albatross"
  },
  {
    "id": "n02066245",
    "superclass": "marine mammals",
    "class": "grey whale"
  },
  {
    "id": "n02071294",
    "superclass": "marine mammals",
    "class": "killer whale"
  },
  {
    "id": "n02074367",
    "superclass": "marine mammals",
    "class": "dugong"
  },
  {
    "id": "n02077923",
    "superclass": "marine mammals",
    "class": "sea lion"
  },
  {
    "id": "n02085620",
    "superclass": "dog",
    "class": "Chihuahua"
  },
  {
    "id": "n02085782",
    "superclass": "dog",
    "class": "Japanese spaniel"
  },
  {
    "id": "n02085936",
    "superclass": "dog",
    "class": "Maltese dog"
  },
  {
    "id": "n02086079",
    "superclass": "dog",
    "class": "Pekinese"
  },
  {
    "id": "n02086240",
    "superclass": "dog",
    "class": "Shih-Tzu"
  },
  {
    "id": "n02086646",
    "superclass": "dog",
    "class": "Blenheim spaniel"
  },
  {
    "id": "n02086910",
    "superclass": "dog",
    "class": "papillon"
  },
  {
    "id": "n02087046",
    "superclass": "dog",
    "class": "toy terrier"
  },
  {
    "id": "n02087394",
    "superclass": "dog",
    "class": "Rhodesian ridgeback"
  },
  {
    "id": "n02088094",
    "superclass": "dog",
    "class": "Afghan hound"
  },
  {
    "id": "n02088238",
    "superclass": "dog",
    "class": "basset"
  },
  {
    "id": "n02088364",
    "superclass": "dog",
    "class": "beagle"
  },
  {
    "id": "n02088466",
    "superclass": "dog",
    "class": "bloodhound"
  },
  {
    "id": "n02088632",
    "superclass": "dog",
    "class": "bluetick"
  },
  {
    "id": "n02089078",
    "superclass": "dog",
    "class": "black-and-tan coonhound"
  },
  {
    "id": "n02089867",
    "superclass": "dog",
    "class": "Walker hound"
  },
  {
    "id": "n02089973",
    "superclass": "dog",
    "class": "English foxhound"
  },
  {
    "id": "n02090379",
    "superclass": "dog",
    "class": "redbone"
  },
  {
    "id": "n02090622",
    "superclass": "dog",
    "class": "borzoi"
  },
  {
    "id": "n02090721",
    "superclass": "dog",
    "class": "Irish wolfhound"
  },
  {
    "id": "n02091032",
    "superclass": "dog",
    "class": "Italian greyhound"
  },
  {
    "id": "n02091134",
    "superclass": "dog",
    "class": "whippet"
  },
  {
    "id": "n02091244",
    "superclass": "dog",
    "class": "Ibizan hound"
  },
  {
    "id": "n02091467",
    "superclass": "dog",
    "class": "Norwegian elkhound"
  },
  {
    "id": "n02091635",
    "superclass": "dog",
    "class": "otterhound"
  },
  {
    "id": "n02091831",
    "superclass": "dog",
    "class": "Saluki"
  },
  {
    "id": "n02092002",
    "superclass": "dog",
    "class": "Scottish deerhound"
  },
  {
    "id": "n02092339",
    "superclass": "dog",
    "class": "Weimaraner"
  },
  {
    "id": "n02093256",
    "superclass": "dog",
    "class": "Staffordshire bullterrier"
  },
  {
    "id": "n02093428",
    "superclass": "dog",
    "class": "American Staffordshire terrier"
  },
  {
    "id": "n02093647",
    "superclass": "dog",
    "class": "Bedlington terrier"
  },
  {
    "id": "n02093754",
    "superclass": "dog",
    "class": "Border terrier"
  },
  {
    "id": "n02093859",
    "superclass": "dog",
    "class": "Kerry blue terrier"
  },
  {
    "id": "n02093991",
    "superclass": "dog",
    "class": "Irish terrier"
  },
  {
    "id": "n02094114",
    "superclass": "dog",
    "class": "Norfolk terrier"
  },
  {
    "id": "n02094258",
    "superclass": "dog",
    "class": "Norwich terrier"
  },
  {
    "id": "n02094433",
    "superclass": "dog",
    "class": "Yorkshire terrier"
  },
  {
    "id": "n02095314",
    "superclass": "dog",
    "class": "wire-haired fox terrier"
  },
  {
    "id": "n02095570",
    "superclass": "dog",
    "class": "Lakeland terrier"
  },
  {
    "id": "n02095889",
    "superclass": "dog",
    "class": "Sealyham terrier"
  },
  {
    "id": "n02096051",
    "superclass": "dog",
    "class": "Airedale"
  },
  {
    "id": "n02096177",
    "superclass": "dog",
    "class": "cairn"
  },
  {
    "id": "n02096294",
    "superclass": "dog",
    "class": "Australian terrier"
  },
  {
    "id": "n02096437",
    "superclass": "dog",
    "class": "Dandie Dinmont"
  },
  {
    "id": "n02096585",
    "superclass": "dog",
    "class": "Boston bull"
  },
  {
    "id": "n02097047",
    "superclass": "dog",
    "class": "miniature schnauzer"
  },
  {
    "id": "n02097130",
    "superclass": "dog",
    "class": "giant schnauzer"
  },
  {
    "id": "n02097209",
    "superclass": "dog",
    "class": "standard schnauzer"
  },
  {
    "id": "n02097298",
    "superclass": "dog",
    "class": "Scotch terrier"
  },
  {
    "id": "n02097474",
    "superclass": "dog",
    "class": "Tibetan terrier"
  },
  {
    "id": "n02097658",
    "superclass": "dog",
    "class": "silky terrier"
  },
  {
    "id": "n02098105",
    "superclass": "dog",
    "class": "soft-coated wheaten terrier"
  },
  {
    "id": "n02098286",
    "superclass": "dog",
    "class": "West Highland white terrier"
  },
  {
    "id": "n02098413",
    "superclass": "dog",
    "class": "Lhasa"
  },
  {
    "id": "n02099267",
    "superclass": "dog",
    "class": "flat-coated retriever"
  },
  {
    "id": "n02099429",
    "superclass": "dog",
    "class": "curly-coated retriever"
  },
  {
    "id": "n02099601",
    "superclass": "dog",
    "class": "golden retriever"
  },
  {
    "id": "n02099712",
    "superclass": "dog",
    "class": "Labrador retriever"
  },
  {
    "id": "n02099849",
    "superclass": "dog",
    "class": "Chesapeake Bay retriever"
  },
  {
    "id": "n02100236",
    "superclass": "dog",
    "class": "German short-haired pointer"
  },
  {
    "id": "n02100583",
    "superclass": "dog",
    "class": "vizsla"
  },
  {
    "id": "n02100735",
    "superclass": "dog",
    "class": "English setter"
  },
  {
    "id": "n02100877",
    "superclass": "dog",
    "class": "Irish setter"
  },
  {
    "id": "n02101006",
    "superclass": "dog",
    "class": "Gordon setter"
  },
  {
    "id": "n02101388",
    "superclass": "dog",
    "class": "Brittany spaniel"
  },
  {
    "id": "n02101556",
    "superclass": "dog",
    "class": "clumber"
  },
  {
    "id": "n02102040",
    "superclass": "dog",
    "class": "English springer"
  },
  {
    "id": "n02102177",
    "superclass": "dog",
    "class": "Welsh springer spaniel"
  },
  {
    "id": "n02102318",
    "superclass": "dog",
    "class": "cocker spaniel"
  },
  {
    "id": "n02102480",
    "superclass": "dog",
    "class": "Sussex spaniel"
  },
  {
    "id": "n02102973",
    "superclass": "dog",
    "class": "Irish water spaniel"
  },
  {
    "id": "n02104029",
    "superclass": "dog",
    "class": "kuvasz"
  },
  {
    "id": "n02104365",
    "superclass": "dog",
    "class": "schipperke"
  },
  {
    "id": "n02105056",
    "superclass": "dog",
    "class": "groenendael"
  },
  {
    "id": "n02105162",
    "superclass": "dog",
    "class": "malinois"
  },
  {
    "id": "n02105251",
    "superclass": "dog",
    "class": "briard"
  },
  {
    "id": "n02105412",
    "superclass": "dog",
    "class": "kelpie"
  },
  {
    "id": "n02105505",
    "superclass": "dog",
    "class": "komondor"
  },
  {
    "id": "n02105641",
    "superclass": "dog",
    "class": "Old English sheepdog"
  },
  {
    "id": "n02105855",
    "superclass": "dog",
    "class": "Shetland sheepdog"
  },
  {
    "id": "n02106030",
    "superclass": "dog",
    "class": "collie"
  },
  {
    "id": "n02106166",
    "superclass": "dog",
    "class": "Border collie"
  },
  {
    "id": "n02106382",
    "superclass": "dog",
    "class": "Bouvier des Flandres"
  },
  {
    "id": "n02106550",
    "superclass": "dog",
    "class": "Rottweiler"
  },
  {
    "id": "n02106662",
    "superclass": "dog",
    "class": "German shepherd"
  },
  {
    "id": "n02107142",
    "superclass": "dog",
    "class": "Doberman"
  },
  {
    "id": "n02107312",
    "superclass": "dog",
    "class": "miniature pinscher"
  },
  {
    "id": "n02107574",
    "superclass": "dog",
    "class": "Greater Swiss Mountain dog"
  },
  {
    "id": "n02107683",
    "superclass": "dog",
    "class": "Bernese mountain dog"
  },
  {
    "id": "n02107908",
    "superclass": "dog",
    "class": "Appenzeller"
  },
  {
    "id": "n02108000",
    "superclass": "dog",
    "class": "EntleBucher"
  },
  {
    "id": "n02108089",
    "superclass": "dog",
    "class": "boxer"
  },
  {
    "id": "n02108422",
    "superclass": "dog",
    "class": "bull mastiff"
  },
  {
    "id": "n02108551",
    "superclass": "dog",
    "class": "Tibetan mastiff"
  },
  {
    "id": "n02108915",
    "superclass": "dog",
    "class": "French bulldog"
  },
  {
    "id": "n02109047",
    "superclass": "dog",
    "class": "Great Dane"
  },
  {
    "id": "n02109525",
    "superclass": "dog",
    "class": "Saint Bernard"
  },
  {
    "id": "n02109961",
    "superclass": "dog",
    "class": "Eskimo dog"
  },
  {
    "id": "n02110063",
    "superclass": "dog",
    "class": "malamute"
  },
  {
    "id": "n02110185",
    "superclass": "dog",
    "class": "Siberian husky"
  },
  {
    "id": "n02110341",
    "superclass": "dog",
    "class": "dalmatian"
  },
  {
    "id": "n02110627",
    "superclass": "dog",
    "class": "affenpinscher"
  },
  {
    "id": "n02110806",
    "superclass": "dog",
    "class": "basenji"
  },
  {
    "id": "n02110958",
    "superclass": "dog",
    "class": "pug"
  },
  {
    "id": "n02111129",
    "superclass": "dog",
    "class": "Leonberg"
  },
  {
    "id": "n02111277",
    "superclass": "dog",
    "class": "Newfoundland"
  },
  {
    "id": "n02111500",
    "superclass": "dog",
    "class": "Great Pyrenees"
  },
  {
    "id": "n02111889",
    "superclass": "dog",
    "class": "Samoyed"
  },
  {
    "id": "n02112018",
    "superclass": "dog",
    "class": "Pomeranian"
  },
  {
    "id": "n02112137",
    "superclass": "dog",
    "class": "chow"
  },
  {
    "id": "n02112350",
    "superclass": "dog",
    "class": "keeshond"
  },
  {
    "id": "n02112706",
    "superclass": "dog",
    "class": "Brabancon griffon"
  },
  {
    "id": "n02113023",
    "superclass": "dog",
    "class": "Pembroke"
  },
  {
    "id": "n02113186",
    "superclass": "dog",
    "class": "Cardigan"
  },
  {
    "id": "n02113624",
    "superclass": "dog",
    "class": "toy poodle"
  },
  {
    "id": "n02113712",
    "superclass": "dog",
    "class": "miniature poodle"
  },
  {
    "id": "n02113799",
    "superclass": "dog",
    "class": "standard poodle"
  },
  {
    "id": "n02113978",
    "superclass": "dog",
    "class": "Mexican hairless"
  },
  {
    "id": "n02114367",
    "superclass": "wild dog",
    "class": "timber wolf"
  },
  {
    "id": "n02114548",
    "superclass": "wild dog",
    "class": "white wolf"
  },
  {
    "id": "n02114712",
    "superclass": "wild dog",
    "class": "red wolf"
  },
  {
    "id": "n02114855",
    "superclass": "wild dog",
    "class": "coyote"
  },
  {
    "id": "n02115641",
    "superclass": "wild dog",
    "class": "dingo"
  },
  {
    "id": "n02115913",
    "superclass": "wild dog",
    "class": "dhole"
  },
  {
    "id": "n02116738",
    "superclass": "dog",
    "class": "African hunting dog"
  },
  {
    "id": "n02117135",
    "superclass": "wild dog",
    "class": "hyena"
  },
  {
    "id": "n02119022",
    "superclass": "wild dog",
    "class": "red fox"
  },
  {
    "id": "n02119789",
    "superclass": "wild dog",
    "class": "kit fox"
  },
  {
    "id": "n02120079",
    "superclass": "wild dog",
    "class": "Arctic fox"
  },
  {
    "id": "n02120505",
    "superclass": "wild dog",
    "class": "grey fox"
  },
  {
    "id": "n02123045",
    "superclass": "cat",
    "class": "tabby"
  },
  {
    "id": "n02123159",
    "superclass": "wild cat",
    "class": "tiger cat"
  },
  {
    "id": "n02123394",
    "superclass": "cat",
    "class": "Persian cat"
  },
  {
    "id": "n02123597",
    "superclass": "cat",
    "class": "Siamese cat"
  },
  {
    "id": "n02124075",
    "superclass": "cat",
    "class": "Egyptian cat"
  },
  {
    "id": "n02125311",
    "superclass": "wild cat",
    "class": "cougar"
  },
  {
    "id": "n02127052",
    "superclass": "wild cat",
    "class": "lynx"
  },
  {
    "id": "n02128385",
    "superclass": "wild cat",
    "class": "leopard"
  },
  {
    "id": "n02128757",
    "superclass": "wild cat",
    "class": "snow leopard"
  },
  {
    "id": "n02128925",
    "superclass": "wild cat",
    "class": "jaguar"
  },
  {
    "id": "n02129165",
    "superclass": "wild cat",
    "class": "lion"
  },
  {
    "id": "n02129604",
    "superclass": "wild cat",
    "class": "tiger"
  },
  {
    "id": "n02130308",
    "superclass": "wild cat",
    "class": "cheetah"
  },
  {
    "id": "n02132136",
    "superclass": "bear",
    "class": "brown bear"
  },
  {
    "id": "n02133161",
    "superclass": "bear",
    "class": "American black bear"
  },
  {
    "id": "n02134084",
    "superclass": "bear",
    "class": "ice bear"
  },
  {
    "id": "n02134418",
    "superclass": "sloth",
    "class": "sloth bear"
  },
  {
    "id": "n02137549",
    "superclass": "mongoose",
    "class": "mongoose"
  },
  {
    "id": "n02138441",
    "superclass": "mongoose",
    "class": "meerkat"
  },
  {
    "id": "n02165105",
    "superclass": "bug",
    "class": "tiger beetle"
  },
  {
    "id": "n02165456",
    "superclass": "bug",
    "class": "ladybug"
  },
  {
    "id": "n02167151",
    "superclass": "bug",
    "class": "ground beetle"
  },
  {
    "id": "n02168699",
    "superclass": "bug",
    "class": "long-horned beetle"
  },
  {
    "id": "n02169497",
    "superclass": "bug",
    "class": "leaf beetle"
  },
  {
    "id": "n02172182",
    "superclass": "bug",
    "class": "dung beetle"
  },
  {
    "id": "n02174001",
    "superclass": "bug",
    "class": "rhinoceros beetle"
  },
  {
    "id": "n02177972",
    "superclass": "bug",
    "class": "weevil"
  },
  {
    "id": "n02190166",
    "superclass": "bug",
    "class": "fly"
  },
  {
    "id": "n02206856",
    "superclass": "bug",
    "class": "bee"
  },
  {
    "id": "n02219486",
    "superclass": "bug",
    "class": "ant"
  },
  {
    "id": "n02226429",
    "superclass": "bug",
    "class": "grasshopper"
  },
  {
    "id": "n02229544",
    "superclass": "bug",
    "class": "cricket"
  },
  {
    "id": "n02231487",
    "superclass": "bug",
    "class": "walking stick"
  },
  {
    "id": "n02233338",
    "superclass": "bug",
    "class": "cockroach"
  },
  {
    "id": "n02236044",
    "superclass": "bug",
    "class": "mantis"
  },
  {
    "id": "n02256656",
    "superclass": "bug",
    "class": "cicada"
  },
  {
    "id": "n02259212",
    "superclass": "bug",
    "class": "leafhopper"
  },
  {
    "id": "n02264363",
    "superclass": "bug",
    "class": "lacewing"
  },
  {
    "id": "n02268443",
    "superclass": "bug",
    "class": "dragonfly"
  },
  {
    "id": "n02268853",
    "superclass": "bug",
    "class": "damselfly"
  },
  {
    "id": "n02276258",
    "superclass": "butterfly",
    "class": "admiral"
  },
  {
    "id": "n02277742",
    "superclass": "butterfly",
    "class": "ringlet"
  },
  {
    "id": "n02279972",
    "superclass": "butterfly",
    "class": "monarch"
  },
  {
    "id": "n02280649",
    "superclass": "butterfly",
    "class": "cabbage butterfly"
  },
  {
    "id": "n02281406",
    "superclass": "butterfly",
    "class": "sulphur butterfly"
  },
  {
    "id": "n02281787",
    "superclass": "butterfly",
    "class": "lycaenid"
  },
  {
    "id": "n02317335",
    "superclass": "echinoderms",
    "class": "starfish"
  },
  {
    "id": "n02319095",
    "superclass": "echinoderms",
    "class": "sea urchin"
  },
  {
    "id": "n02321529",
    "superclass": "echinoderms",
    "class": "sea cucumber"
  },
  {
    "id": "n02325366",
    "superclass": "rabbit",
    "class": "wood rabbit"
  },
  {
    "id": "n02326432",
    "superclass": "rabbit",
    "class": "hare"
  },
  {
    "id": "n02328150",
    "superclass": "rabbit",
    "class": "Angora"
  },
  {
    "id": "n02342885",
    "superclass": "rodent",
    "class": "hamster"
  },
  {
    "id": "n02346627",
    "superclass": "rodent",
    "class": "porcupine"
  },
  {
    "id": "n02356798",
    "superclass": "rodent",
    "class": "fox squirrel"
  },
  {
    "id": "n02361337",
    "superclass": "rodent",
    "class": "marmot"
  },
  {
    "id": "n02363005",
    "superclass": "rodent",
    "class": "beaver"
  },
  {
    "id": "n02364673",
    "superclass": "rodent",
    "class": "guinea pig"
  },
  {
    "id": "n02389026",
    "superclass": "horse",
    "class": "sorrel"
  },
  {
    "id": "n02391049",
    "superclass": "ungulate",
    "class": "zebra"
  },
  {
    "id": "n02395406",
    "superclass": "hog",
    "class": "hog"
  },
  {
    "id": "n02396427",
    "superclass": "hog",
    "class": "wild boar"
  },
  {
    "id": "n02397096",
    "superclass": "hog",
    "class": "warthog"
  },
  {
    "id": "n02398521",
    "superclass": "ungulate",
    "class": "hippopotamus"
  },
  {
    "id": "n02403003",
    "superclass": "ungulate",
    "class": "ox"
  },
  {
    "id": "n02408429",
    "superclass": "ungulate",
    "class": "water buffalo"
  },
  {
    "id": "n02410509",
    "superclass": "ungulate",
    "class": "bison"
  },
  {
    "id": "n02412080",
    "superclass": "ungulate",
    "class": "ram"
  },
  {
    "id": "n02415577",
    "superclass": "ungulate",
    "class": "bighorn"
  },
  {
    "id": "n02417914",
    "superclass": "ungulate",
    "class": "ibex"
  },
  {
    "id": "n02422106",
    "superclass": "ungulate",
    "class": "hartebeest"
  },
  {
    "id": "n02422699",
    "superclass": "ungulate",
    "class": "impala"
  },
  {
    "id": "n02423022",
    "superclass": "ungulate",
    "class": "gazelle"
  },
  {
    "id": "n02437312",
    "superclass": "ungulate",
    "class": "Arabian camel"
  },
  {
    "id": "n02437616",
    "superclass": "ungulate",
    "class": "llama"
  },
  {
    "id": "n02441942",
    "superclass": "ferret",
    "class": "weasel"
  },
  {
    "id": "n02442845",
    "superclass": "ferret",
    "class": "mink"
  },
  {
    "id": "n02443114",
    "superclass": "animal",
    "class": "polecat"
  },
  {
    "id": "n02443484",
    "superclass": "ferret",
    "class": "black-footed ferret"
  },
  {
    "id": "n02444819",
    "superclass": "ferret",
    "class": "otter"
  },
  {
    "id": "n02445715",
    "superclass": "ferret",
    "class": "skunk"
  },
  {
    "id": "n02447366",
    "superclass": "ferret",
    "class": "badger"
  },
  {
    "id": "n02454379",
    "superclass": "armadillo",
    "class": "armadillo"
  },
  {
    "id": "n02457408",
    "superclass": "sloth",
    "class": "three-toed sloth"
  },
  {
    "id": "n02480495",
    "superclass": "primate",
    "class": "orangutan"
  },
  {
    "id": "n02480855",
    "superclass": "primate",
    "class": "gorilla"
  },
  {
    "id": "n02481823",
    "superclass": "primate",
    "class": "chimpanzee"
  },
  {
    "id": "n02483362",
    "superclass": "primate",
    "class": "gibbon"
  },
  {
    "id": "n02483708",
    "superclass": "primate",
    "class": "siamang"
  },
  {
    "id": "n02484975",
    "superclass": "primate",
    "class": "guenon"
  },
  {
    "id": "n02486261",
    "superclass": "primate",
    "class": "patas"
  },
  {
    "id": "n02486410",
    "superclass": "primate",
    "class": "baboon"
  },
  {
    "id": "n02487347",
    "superclass": "primate",
    "class": "macaque"
  },
  {
    "id": "n02488291",
    "superclass": "primate",
    "class": "langur"
  },
  {
    "id": "n02488702",
    "superclass": "primate",
    "class": "colobus"
  },
  {
    "id": "n02489166",
    "superclass": "primate",
    "class": "proboscis monkey"
  },
  {
    "id": "n02490219",
    "superclass": "primate",
    "class": "marmoset"
  },
  {
    "id": "n02492035",
    "superclass": "primate",
    "class": "capuchin"
  },
  {
    "id": "n02492660",
    "superclass": "primate",
    "class": "howler monkey"
  },
  {
    "id": "n02493509",
    "superclass": "primate",
    "class": "titi"
  },
  {
    "id": "n02493793",
    "superclass": "primate",
    "class": "spider monkey"
  },
  {
    "id": "n02494079",
    "superclass": "primate",
    "class": "squirrel monkey"
  },
  {
    "id": "n02497673",
    "superclass": "primate",
    "class": "Madagascar cat"
  },
  {
    "id": "n02500267",
    "superclass": "primate",
    "class": "indri"
  },
  {
    "id": "n02504013",
    "superclass": "ungulate",
    "class": "Indian elephant"
  },
  {
    "id": "n02504458",
    "superclass": "ungulate",
    "class": "African elephant"
  },
  {
    "id": "n02509815",
    "superclass": "bear",
    "class": "lesser panda"
  },
  {
    "id": "n02510455",
    "superclass": "bear",
    "class": "giant panda"
  },
  {
    "id": "n02514041",
    "superclass": "fish",
    "class": "barracouta"
  },
  {
    "id": "n02526121",
    "superclass": "fish",
    "class": "eel"
  },
  {
    "id": "n02536864",
    "superclass": "fish",
    "class": "coho"
  },
  {
    "id": "n02606052",
    "superclass": "fish",
    "class": "rock beauty"
  },
  {
    "id": "n02607072",
    "superclass": "fish",
    "class": "anemone fish"
  },
  {
    "id": "n02640242",
    "superclass": "fish",
    "class": "sturgeon"
  },
  {
    "id": "n02641379",
    "superclass": "fish",
    "class": "gar"
  },
  {
    "id": "n02643566",
    "superclass": "fish",
    "class": "lionfish"
  },
  {
    "id": "n02655020",
    "superclass": "fish",
    "class": "puffer"
  },
  {
    "id": "n02666196",
    "superclass": "technology",
    "class": "abacus"
  },
  {
    "id": "n02667093",
    "superclass": "clothing",
    "class": "abaya"
  },
  {
    "id": "n02669723",
    "superclass": "clothing",
    "class": "academic gown"
  },
  {
    "id": "n02672831",
    "superclass": "instrument",
    "class": "accordion"
  },
  {
    "id": "n02676566",
    "superclass": "instrument",
    "class": "acoustic guitar"
  },
  {
    "id": "n02687172",
    "superclass": "boat",
    "class": "aircraft carrier"
  },
  {
    "id": "n02690373",
    "superclass": "aircraft",
    "class": "airliner"
  },
  {
    "id": "n02692877",
    "superclass": "aircraft",
    "class": "airship"
  },
  {
    "id": "n02699494",
    "superclass": "furniture",
    "class": "altar"
  },
  {
    "id": "n02701002",
    "superclass": "vehicle",
    "class": "ambulance"
  },
  {
    "id": "n02704792",
    "superclass": "vehicle",
    "class": "amphibian"
  },
  {
    "id": "n02708093",
    "superclass": "decor",
    "class": "analog clock"
  },
  {
    "id": "n02727426",
    "superclass": "outdoor scene",
    "class": "apiary"
  },
  {
    "id": "n02730930",
    "superclass": "clothing",
    "class": "apron"
  },
  {
    "id": "n02747177",
    "superclass": "furniture",
    "class": "ashcan"
  },
  {
    "id": "n02749479",
    "superclass": "weapon",
    "class": "assault rifle"
  },
  {
    "id": "n02769748",
    "superclass": "accessory",
    "class": "backpack"
  },
  {
    "id": "n02776631",
    "superclass": "food",
    "class": "bakery"
  },
  {
    "id": "n02777292",
    "superclass": "sports equipment",
    "class": "balance beam"
  },
  {
    "id": "n02782093",
    "superclass": "toy",
    "class": "balloon"
  },
  {
    "id": "n02783161",
    "superclass": "tool",
    "class": "ballpoint"
  },
  {
    "id": "n02786058",
    "superclass": "wound care product",
    "class": "Band Aid"
  },
  {
    "id": "n02787622",
    "superclass": "instrument",
    "class": "banjo"
  },
  {
    "id": "n02788148",
    "superclass": "furniture",
    "class": "bannister"
  },
  {
    "id": "n02790996",
    "superclass": "fitness equipment",
    "class": "barbell"
  },
  {
    "id": "n02791124",
    "superclass": "furniture",
    "class": "barber chair"
  },
  {
    "id": "n02791270",
    "superclass": "place",
    "class": "barbershop"
  },
  {
    "id": "n02793495",
    "superclass": "building",
    "class": "barn"
  },
  {
    "id": "n02794156",
    "superclass": "technology",
    "class": "barometer"
  },
  {
    "id": "n02795169",
    "superclass": "container",
    "class": "barrel"
  },
  {
    "id": "n02797295",
    "superclass": "tool",
    "class": "barrow"
  },
  {
    "id": "n02799071",
    "superclass": "ball",
    "class": "baseball"
  },
  {
    "id": "n02802426",
    "superclass": "ball",
    "class": "basketball"
  },
  {
    "id": "n02804414",
    "superclass": "furniture",
    "class": "bassinet"
  },
  {
    "id": "n02804610",
    "superclass": "instrument",
    "class": "bassoon"
  },
  {
    "id": "n02807133",
    "superclass": "hat",
    "class": "bathing cap"
  },
  {
    "id": "n02808304",
    "superclass": "bathroom supply",
    "class": "bath towel"
  },
  {
    "id": "n02808440",
    "superclass": "bathroom fixture",
    "class": "bathtub"
  },
  {
    "id": "n02814533",
    "superclass": "vehicle",
    "class": "beach wagon"
  },
  {
    "id": "n02814860",
    "superclass": "safety equipment",
    "class": "beacon"
  },
  {
    "id": "n02815834",
    "superclass": "lab equipment",
    "class": "beaker"
  },
  {
    "id": "n02817516",
    "superclass": "hat",
    "class": "bearskin"
  },
  {
    "id": "n02823428",
    "superclass": "food",
    "class": "beer bottle"
  },
  {
    "id": "n02823750",
    "superclass": "food",
    "class": "beer glass"
  },
  {
    "id": "n02825657",
    "superclass": "building",
    "class": "bell cote"
  },
  {
    "id": "n02834397",
    "superclass": "clothing",
    "class": "bib"
  },
  {
    "id": "n02835271",
    "superclass": "vehicle",
    "class": "bicycle-built-for-two"
  },
  {
    "id": "n02837789",
    "superclass": "clothing",
    "class": "bikini"
  },
  {
    "id": "n02840245",
    "superclass": "paper",
    "class": "binder"
  },
  {
    "id": "n02841315",
    "superclass": "tool",
    "class": "binoculars"
  },
  {
    "id": "n02843684",
    "superclass": "outdoor scene",
    "class": "birdhouse"
  },
  {
    "id": "n02859443",
    "superclass": "building",
    "class": "boathouse"
  },
  {
    "id": "n02860847",
    "superclass": "vehicle",
    "class": "bobsled"
  },
  {
    "id": "n02865351",
    "superclass": "accessory",
    "class": "bolo tie"
  },
  {
    "id": "n02869837",
    "superclass": "hat",
    "class": "bonnet"
  },
  {
    "id": "n02870880",
    "superclass": "furniture",
    "class": "bookcase"
  },
  {
    "id": "n02871525",
    "superclass": "building",
    "class": "bookshop"
  },
  {
    "id": "n02877765",
    "superclass": "packaging material",
    "class": "bottlecap"
  },
  {
    "id": "n02879718",
    "superclass": "weapon",
    "class": "bow"
  },
  {
    "id": "n02883205",
    "superclass": "accessory",
    "class": "bow tie"
  },
  {
    "id": "n02892201",
    "superclass": "decor",
    "class": "brass"
  },
  {
    "id": "n02892767",
    "superclass": "clothing",
    "class": "brassiere"
  },
  {
    "id": "n02894605",
    "superclass": "outdoor scene",
    "class": "breakwater"
  },
  {
    "id": "n02895154",
    "superclass": "clothing",
    "class": "breastplate"
  },
  {
    "id": "n02906734",
    "superclass": "tool",
    "class": "broom"
  },
  {
    "id": "n02909870",
    "superclass": "tool",
    "class": "bucket"
  },
  {
    "id": "n02910353",
    "superclass": "accessory",
    "class": "buckle"
  },
  {
    "id": "n02916936",
    "superclass": "clothing",
    "class": "bulletproof vest"
  },
  {
    "id": "n02917067",
    "superclass": "train",
    "class": "bullet train"
  },
  {
    "id": "n02927161",
    "superclass": "place",
    "class": "butcher shop"
  },
  {
    "id": "n02930766",
    "superclass": "vehicle",
    "class": "cab"
  },
  {
    "id": "n02939185",
    "superclass": "kitchenware",
    "class": "caldron"
  },
  {
    "id": "n02948072",
    "superclass": "tool",
    "class": "candle"
  },
  {
    "id": "n02950826",
    "superclass": "weapon",
    "class": "cannon"
  },
  {
    "id": "n02951358",
    "superclass": "boat",
    "class": "canoe"
  },
  {
    "id": "n02951585",
    "superclass": "tool",
    "class": "can opener"
  },
  {
    "id": "n02963159",
    "superclass": "clothing",
    "class": "cardigan"
  },
  {
    "id": "n02965783",
    "superclass": "technology",
    "class": "car mirror"
  },
  {
    "id": "n02966193",
    "superclass": "ride",
    "class": "carousel"
  },
  {
    "id": "n02966687",
    "superclass": "tool",
    "class": "carpenter's kit"
  },
  {
    "id": "n02971356",
    "superclass": "container",
    "class": "carton"
  },
  {
    "id": "n02974003",
    "superclass": "technology",
    "class": "car wheel"
  },
  {
    "id": "n02977058",
    "superclass": "technology",
    "class": "cash machine"
  },
  {
    "id": "n02978881",
    "superclass": "technology",
    "class": "cassette"
  },
  {
    "id": "n02979186",
    "superclass": "electronics",
    "class": "cassette player"
  },
  {
    "id": "n02980441",
    "superclass": "building",
    "class": "castle"
  },
  {
    "id": "n02981792",
    "superclass": "boat",
    "class": "catamaran"
  },
  {
    "id": "n02988304",
    "superclass": "electronics",
    "class": "CD player"
  },
  {
    "id": "n02992211",
    "superclass": "instrument",
    "class": "cello"
  },
  {
    "id": "n02992529",
    "superclass": "electronics",
    "class": "cellular telephone"
  },
  {
    "id": "n02999410",
    "superclass": "tool",
    "class": "chain"
  },
  {
    "id": "n03000134",
    "superclass": "fence",
    "class": "chainlink fence"
  },
  {
    "id": "n03000247",
    "superclass": "clothing",
    "class": "chain mail"
  },
  {
    "id": "n03000684",
    "superclass": "tool",
    "class": "chain saw"
  },
  {
    "id": "n03014705",
    "superclass": "container",
    "class": "chest"
  },
  {
    "id": "n03016953",
    "superclass": "furniture",
    "class": "chiffonier"
  },
  {
    "id": "n03017168",
    "superclass": "instrument",
    "class": "chime"
  },
  {
    "id": "n03018349",
    "superclass": "furniture",
    "class": "china cabinet"
  },
  {
    "id": "n03026506",
    "superclass": "decor",
    "class": "Christmas stocking"
  },
  {
    "id": "n03028079",
    "superclass": "building",
    "class": "church"
  },
  {
    "id": "n03032252",
    "superclass": "place",
    "class": "cinema"
  },
  {
    "id": "n03041632",
    "superclass": "cooking",
    "class": "cleaver"
  },
  {
    "id": "n03042490",
    "superclass": "building",
    "class": "cliff dwelling"
  },
  {
    "id": "n03045698",
    "superclass": "clothing",
    "class": "cloak"
  },
  {
    "id": "n03047690",
    "superclass": "clothing",
    "class": "clog"
  },
  {
    "id": "n03062245",
    "superclass": "cooking",
    "class": "cocktail shaker"
  },
  {
    "id": "n03063599",
    "superclass": "kitchenware",
    "class": "coffee mug"
  },
  {
    "id": "n03063689",
    "superclass": "kitchenware",
    "class": "coffeepot"
  },
  {
    "id": "n03065424",
    "superclass": "technology",
    "class": "coil"
  },
  {
    "id": "n03075370",
    "superclass": "technology",
    "class": "combination lock"
  },
  {
    "id": "n03085013",
    "superclass": "electronics",
    "class": "computer keyboard"
  },
  {
    "id": "n03089624",
    "superclass": "food",
    "class": "confectionery"
  },
  {
    "id": "n03095699",
    "superclass": "boat",
    "class": "container ship"
  },
  {
    "id": "n03100240",
    "superclass": "vehicle",
    "class": "convertible"
  },
  {
    "id": "n03109150",
    "superclass": "tool",
    "class": "corkscrew"
  },
  {
    "id": "n03110669",
    "superclass": "instrument",
    "class": "cornet"
  },
  {
    "id": "n03124043",
    "superclass": "clothing",
    "class": "cowboy boot"
  },
  {
    "id": "n03124170",
    "superclass": "clothing",
    "class": "cowboy hat"
  },
  {
    "id": "n03125729",
    "superclass": "furniture",
    "class": "cradle"
  },
  {
    "id": "n03126707",
    "superclass": "vehicle",
    "class": "crane"
  },
  {
    "id": "n03127747",
    "superclass": "hat",
    "class": "crash helmet"
  },
  {
    "id": "n03127925",
    "superclass": "container",
    "class": "crate"
  },
  {
    "id": "n03131574",
    "superclass": "furniture",
    "class": "crib"
  },
  {
    "id": "n03133878",
    "superclass": "cooking",
    "class": "Crock Pot"
  },
  {
    "id": "n03134739",
    "superclass": "ball",
    "class": "croquet ball"
  },
  {
    "id": "n03141823",
    "superclass": "tool",
    "class": "crutch"
  },
  {
    "id": "n03146219",
    "superclass": "clothing",
    "class": "cuirass"
  },
  {
    "id": "n03160309",
    "superclass": "outdoor scene",
    "class": "dam"
  },
  {
    "id": "n03179701",
    "superclass": "furniture",
    "class": "desk"
  },
  {
    "id": "n03180011",
    "superclass": "electronics",
    "class": "desktop computer"
  },
  {
    "id": "n03187595",
    "superclass": "electronics",
    "class": "dial telephone"
  },
  {
    "id": "n03188531",
    "superclass": "clothing",
    "class": "diaper"
  },
  {
    "id": "n03196217",
    "superclass": "electronics",
    "class": "digital clock"
  },
  {
    "id": "n03197337",
    "superclass": "electronics",
    "class": "digital watch"
  },
  {
    "id": "n03201208",
    "superclass": "furniture",
    "class": "dining table"
  },
  {
    "id": "n03207743",
    "superclass": "decor",
    "class": "dishrag"
  },
  {
    "id": "n03207941",
    "superclass": "home appliance",
    "class": "dishwasher"
  },
  {
    "id": "n03208938",
    "superclass": "technology",
    "class": "disk brake"
  },
  {
    "id": "n03216828",
    "superclass": "outdoor scene",
    "class": "dock"
  },
  {
    "id": "n03218198",
    "superclass": "vehicle",
    "class": "dogsled"
  },
  {
    "id": "n03220513",
    "superclass": "building",
    "class": "dome"
  },
  {
    "id": "n03223299",
    "superclass": "decor",
    "class": "doormat"
  },
  {
    "id": "n03240683",
    "superclass": "outdoor scene",
    "class": "drilling platform"
  },
  {
    "id": "n03249569",
    "superclass": "instrument",
    "class": "drum"
  },
  {
    "id": "n03250847",
    "superclass": "instrument",
    "class": "drumstick"
  },
  {
    "id": "n03255030",
    "superclass": "tool",
    "class": "dumbbell"
  },
  {
    "id": "n03259280",
    "superclass": "cooking",
    "class": "Dutch oven"
  },
  {
    "id": "n03271574",
    "superclass": "electronics",
    "class": "electric fan"
  },
  {
    "id": "n03272010",
    "superclass": "instrument",
    "class": "electric guitar"
  },
  {
    "id": "n03272562",
    "superclass": "train",
    "class": "electric locomotive"
  },
  {
    "id": "n03290653",
    "superclass": "furniture",
    "class": "entertainment center"
  },
  {
    "id": "n03291819",
    "superclass": "paper",
    "class": "envelope"
  },
  {
    "id": "n03297495",
    "superclass": "cooking",
    "class": "espresso maker"
  },
  {
    "id": "n03314780",
    "superclass": "cosmetics",
    "class": "face powder"
  },
  {
    "id": "n03325584",
    "superclass": "accessory",
    "class": "feather boa"
  },
  {
    "id": "n03337140",
    "superclass": "furniture",
    "class": "file"
  },
  {
    "id": "n03344393",
    "superclass": "boat",
    "class": "fireboat"
  },
  {
    "id": "n03345487",
    "superclass": "vehicle",
    "class": "fire engine"
  },
  {
    "id": "n03347037",
    "superclass": "furniture",
    "class": "fire screen"
  },
  {
    "id": "n03355925",
    "superclass": "outdoor scene",
    "class": "flagpole"
  },
  {
    "id": "n03372029",
    "superclass": "instrument",
    "class": "flute"
  },
  {
    "id": "n03376595",
    "superclass": "furniture",
    "class": "folding chair"
  },
  {
    "id": "n03379051",
    "superclass": "sports equipment",
    "class": "football helmet"
  },
  {
    "id": "n03384352",
    "superclass": "vehicle",
    "class": "forklift"
  },
  {
    "id": "n03388043",
    "superclass": "outdoor scene",
    "class": "fountain"
  },
  {
    "id": "n03388183",
    "superclass": "tool",
    "class": "fountain pen"
  },
  {
    "id": "n03388549",
    "superclass": "furniture",
    "class": "four-poster"
  },
  {
    "id": "n03393912",
    "superclass": "train",
    "class": "freight car"
  },
  {
    "id": "n03394916",
    "superclass": "instrument",
    "class": "French horn"
  },
  {
    "id": "n03400231",
    "superclass": "kitchenware",
    "class": "frying pan"
  },
  {
    "id": "n03404251",
    "superclass": "clothing",
    "class": "fur coat"
  },
  {
    "id": "n03417042",
    "superclass": "vehicle",
    "class": "garbage truck"
  },
  {
    "id": "n03424325",
    "superclass": "accessory",
    "class": "gasmask"
  },
  {
    "id": "n03425413",
    "superclass": "technology",
    "class": "gas pump"
  },
  {
    "id": "n03443371",
    "superclass": "cooking",
    "class": "goblet"
  },
  {
    "id": "n03444034",
    "superclass": "vehicle",
    "class": "go-kart"
  },
  {
    "id": "n03445777",
    "superclass": "ball",
    "class": "golf ball"
  },
  {
    "id": "n03445924",
    "superclass": "vehicle",
    "class": "golf cart"
  },
  {
    "id": "n03447447",
    "superclass": "boat",
    "class": "gondola"
  },
  {
    "id": "n03447721",
    "superclass": "instrument",
    "class": "gong"
  },
  {
    "id": "n03450230",
    "superclass": "clothing",
    "class": "gown"
  },
  {
    "id": "n03452741",
    "superclass": "instrument",
    "class": "grand piano"
  },
  {
    "id": "n03457902",
    "superclass": "building",
    "class": "greenhouse"
  },
  {
    "id": "n03459775",
    "superclass": "vehicle",
    "class": "grille"
  },
  {
    "id": "n03461385",
    "superclass": "place",
    "class": "grocery store"
  },
  {
    "id": "n03467068",
    "superclass": "weapon",
    "class": "guillotine"
  },
  {
    "id": "n03476684",
    "superclass": "accessory",
    "class": "hair slide"
  },
  {
    "id": "n03476991",
    "superclass": "hair styling product",
    "class": "hair spray"
  },
  {
    "id": "n03478589",
    "superclass": "vehicle",
    "class": "half track"
  },
  {
    "id": "n03481172",
    "superclass": "tool",
    "class": "hammer"
  },
  {
    "id": "n03482405",
    "superclass": "decor",
    "class": "hamper"
  },
  {
    "id": "n03483316",
    "superclass": "electronics",
    "class": "hand blower"
  },
  {
    "id": "n03485407",
    "superclass": "electronics",
    "class": "hand-held computer"
  },
  {
    "id": "n03485794",
    "superclass": "accessory",
    "class": "handkerchief"
  },
  {
    "id": "n03492542",
    "superclass": "electronics",
    "class": "hard disc"
  },
  {
    "id": "n03494278",
    "superclass": "instrument",
    "class": "harmonica"
  },
  {
    "id": "n03495258",
    "superclass": "instrument",
    "class": "harp"
  },
  {
    "id": "n03496892",
    "superclass": "vehicle",
    "class": "harvester"
  },
  {
    "id": "n03498962",
    "superclass": "tool",
    "class": "hatchet"
  },
  {
    "id": "n03527444",
    "superclass": "weapon",
    "class": "holster"
  },
  {
    "id": "n03529860",
    "superclass": "place",
    "class": "home theater"
  },
  {
    "id": "n03530642",
    "superclass": "nature",
    "class": "honeycomb"
  },
  {
    "id": "n03532672",
    "superclass": "tool",
    "class": "hook"
  },
  {
    "id": "n03534580",
    "superclass": "clothing",
    "class": "hoopskirt"
  },
  {
    "id": "n03535780",
    "superclass": "sports equipment",
    "class": "horizontal bar"
  },
  {
    "id": "n03538406",
    "superclass": "vehicle",
    "class": "horse cart"
  },
  {
    "id": "n03544143",
    "superclass": "technology",
    "class": "hourglass"
  },
  {
    "id": "n03584254",
    "superclass": "electronics",
    "class": "iPod"
  },
  {
    "id": "n03584829",
    "superclass": "tool",
    "class": "iron"
  },
  {
    "id": "n03590841",
    "superclass": "fruit",
    "class": "jack-o'-lantern"
  },
  {
    "id": "n03594734",
    "superclass": "clothing",
    "class": "jean"
  },
  {
    "id": "n03594945",
    "superclass": "vehicle",
    "class": "jeep"
  },
  {
    "id": "n03595614",
    "superclass": "clothing",
    "class": "jersey"
  },
  {
    "id": "n03598930",
    "superclass": "paper",
    "class": "jigsaw puzzle"
  },
  {
    "id": "n03599486",
    "superclass": "vehicle",
    "class": "jinrikisha"
  },
  {
    "id": "n03602883",
    "superclass": "electronics",
    "class": "joystick"
  },
  {
    "id": "n03617480",
    "superclass": "clothing",
    "class": "kimono"
  },
  {
    "id": "n03623198",
    "superclass": "clothing",
    "class": "knee pad"
  },
  {
    "id": "n03627232",
    "superclass": "tool",
    "class": "knot"
  },
  {
    "id": "n03630383",
    "superclass": "clothing",
    "class": "lab coat"
  },
  {
    "id": "n03633091",
    "superclass": "kitchenware",
    "class": "ladle"
  },
  {
    "id": "n03637318",
    "superclass": "decor",
    "class": "lampshade"
  },
  {
    "id": "n03642806",
    "superclass": "electronics",
    "class": "laptop"
  },
  {
    "id": "n03649909",
    "superclass": "vehicle",
    "class": "lawn mower"
  },
  {
    "id": "n03657121",
    "superclass": "technology",
    "class": "lens cap"
  },
  {
    "id": "n03658185",
    "superclass": "tool",
    "class": "letter opener"
  },
  {
    "id": "n03661043",
    "superclass": "building",
    "class": "library"
  },
  {
    "id": "n03662601",
    "superclass": "boat",
    "class": "lifeboat"
  },
  {
    "id": "n03666591",
    "superclass": "tool",
    "class": "lighter"
  },
  {
    "id": "n03670208",
    "superclass": "vehicle",
    "class": "limousine"
  },
  {
    "id": "n03673027",
    "superclass": "boat",
    "class": "liner"
  },
  {
    "id": "n03676483",
    "superclass": "accessory",
    "class": "lipstick"
  },
  {
    "id": "n03680355",
    "superclass": "clothing",
    "class": "Loafer"
  },
  {
    "id": "n03690938",
    "superclass": "skin care product",
    "class": "lotion"
  },
  {
    "id": "n03691459",
    "superclass": "electronics",
    "class": "loudspeaker"
  },
  {
    "id": "n03692522",
    "superclass": "tool",
    "class": "loupe"
  },
  {
    "id": "n03697007",
    "superclass": "building",
    "class": "lumbermill"
  },
  {
    "id": "n03706229",
    "superclass": "tool",
    "class": "magnetic compass"
  },
  {
    "id": "n03709823",
    "superclass": "accessory",
    "class": "mailbag"
  },
  {
    "id": "n03710193",
    "superclass": "container",
    "class": "mailbox"
  },
  {
    "id": "n03710637",
    "superclass": "clothing",
    "class": "maillot"
  },
  {
    "id": "n03710721",
    "superclass": "clothing",
    "class": "tank suit"
  },
  {
    "id": "n03717622",
    "superclass": "outdoor scene",
    "class": "manhole cover"
  },
  {
    "id": "n03720891",
    "superclass": "instrument",
    "class": "maraca"
  },
  {
    "id": "n03721384",
    "superclass": "instrument",
    "class": "marimba"
  },
  {
    "id": "n03724870",
    "superclass": "clothing",
    "class": "mask"
  },
  {
    "id": "n03729826",
    "superclass": "tool",
    "class": "matchstick"
  },
  {
    "id": "n03733131",
    "superclass": "outdoor scene",
    "class": "maypole"
  },
  {
    "id": "n03733281",
    "superclass": "outdoor scene",
    "class": "maze"
  },
  {
    "id": "n03733805",
    "superclass": "cooking",
    "class": "measuring cup"
  },
  {
    "id": "n03742115",
    "superclass": "container",
    "class": "medicine chest"
  },
  {
    "id": "n03743016",
    "superclass": "outdoor scene",
    "class": "megalith"
  },
  {
    "id": "n03759954",
    "superclass": "kitchenware",
    "class": "microphone"
  },
  {
    "id": "n03761084",
    "superclass": "electronics",
    "class": "microwave"
  },
  {
    "id": "n03763968",
    "superclass": "clothing",
    "class": "military uniform"
  },
  {
    "id": "n03764736",
    "superclass": "container",
    "class": "milk can"
  },
  {
    "id": "n03769881",
    "superclass": "vehicle",
    "class": "minibus"
  },
  {
    "id": "n03770439",
    "superclass": "clothing",
    "class": "miniskirt"
  },
  {
    "id": "n03770679",
    "superclass": "vehicle",
    "class": "minivan"
  },
  {
    "id": "n03773504",
    "superclass": "weapon",
    "class": "missile"
  },
  {
    "id": "n03775071",
    "superclass": "clothing",
    "class": "mitten"
  },
  {
    "id": "n03775546",
    "superclass": "kitchenware",
    "class": "mixing bowl"
  },
  {
    "id": "n03776460",
    "superclass": "building",
    "class": "mobile home"
  },
  {
    "id": "n03777568",
    "superclass": "vehicle",
    "class": "Model T"
  },
  {
    "id": "n03777754",
    "superclass": "electronics",
    "class": "modem"
  },
  {
    "id": "n03781244",
    "superclass": "building",
    "class": "monastery"
  },
  {
    "id": "n03782006",
    "superclass": "electronics",
    "class": "monitor"
  },
  {
    "id": "n03785016",
    "superclass": "vehicle",
    "class": "moped"
  },
  {
    "id": "n03786901",
    "superclass": "kitchenware",
    "class": "mortar"
  },
  {
    "id": "n03787032",
    "superclass": "hat",
    "class": "mortarboard"
  },
  {
    "id": "n03788195",
    "superclass": "building",
    "class": "mosque"
  },
  {
    "id": "n03788365",
    "superclass": "tool",
    "class": "mosquito net"
  },
  {
    "id": "n03791053",
    "superclass": "vehicle",
    "class": "motor scooter"
  },
  {
    "id": "n03792782",
    "superclass": "vehicle",
    "class": "mountain bike"
  },
  {
    "id": "n03792972",
    "superclass": "building",
    "class": "mountain tent"
  },
  {
    "id": "n03793489",
    "superclass": "electronics",
    "class": "mouse"
  },
  {
    "id": "n03794056",
    "superclass": "technology",
    "class": "mousetrap"
  },
  {
    "id": "n03796401",
    "superclass": "vehicle",
    "class": "moving van"
  },
  {
    "id": "n03803284",
    "superclass": "tool",
    "class": "muzzle"
  },
  {
    "id": "n03804744",
    "superclass": "tool",
    "class": "nail"
  },
  {
    "id": "n03814639",
    "superclass": "accessory",
    "class": "neck brace"
  },
  {
    "id": "n03814906",
    "superclass": "accessory",
    "class": "necklace"
  },
  {
    "id": "n03825788",
    "superclass": "child care product",
    "class": "nipple"
  },
  {
    "id": "n03832673",
    "superclass": "electronics",
    "class": "notebook"
  },
  {
    "id": "n03837869",
    "superclass": "building",
    "class": "obelisk"
  },
  {
    "id": "n03838899",
    "superclass": "instrument",
    "class": "oboe"
  },
  {
    "id": "n03840681",
    "superclass": "instrument",
    "class": "ocarina"
  },
  {
    "id": "n03841143",
    "superclass": "electronics",
    "class": "odometer"
  },
  {
    "id": "n03843555",
    "superclass": "technology",
    "class": "oil filter"
  },
  {
    "id": "n03854065",
    "superclass": "instrument",
    "class": "organ"
  },
  {
    "id": "n03857828",
    "superclass": "electronics",
    "class": "oscilloscope"
  },
  {
    "id": "n03866082",
    "superclass": "clothing",
    "class": "overskirt"
  },
  {
    "id": "n03868242",
    "superclass": "vehicle",
    "class": "oxcart"
  },
  {
    "id": "n03868863",
    "superclass": "clothing",
    "class": "oxygen mask"
  },
  {
    "id": "n03871628",
    "superclass": "container",
    "class": "packet"
  },
  {
    "id": "n03873416",
    "superclass": "sports equipment",
    "class": "paddle"
  },
  {
    "id": "n03874293",
    "superclass": "boat",
    "class": "paddlewheel"
  },
  {
    "id": "n03874599",
    "superclass": "technology",
    "class": "padlock"
  },
  {
    "id": "n03876231",
    "superclass": "tool",
    "class": "paintbrush"
  },
  {
    "id": "n03877472",
    "superclass": "clothing",
    "class": "pajama"
  },
  {
    "id": "n03877845",
    "superclass": "building",
    "class": "palace"
  },
  {
    "id": "n03884397",
    "superclass": "instrument",
    "class": "panpipe"
  },
  {
    "id": "n03887697",
    "superclass": "paper",
    "class": "paper towel"
  },
  {
    "id": "n03888257",
    "superclass": "aircraft",
    "class": "parachute"
  },
  {
    "id": "n03888605",
    "superclass": "sports equipment",
    "class": "parallel bars"
  },
  {
    "id": "n03891251",
    "superclass": "furniture",
    "class": "park bench"
  },
  {
    "id": "n03891332",
    "superclass": "electronics",
    "class": "parking meter"
  },
  {
    "id": "n03895866",
    "superclass": "vehicle",
    "class": "passenger car"
  },
  {
    "id": "n03899768",
    "superclass": "place",
    "class": "patio"
  },
  {
    "id": "n03902125",
    "superclass": "electronics",
    "class": "pay-phone"
  },
  {
    "id": "n03903868",
    "superclass": "architectural component",
    "class": "pedestal"
  },
  {
    "id": "n03908618",
    "superclass": "container",
    "class": "pencil box"
  },
  {
    "id": "n03908714",
    "superclass": "tool",
    "class": "pencil sharpener"
  },
  {
    "id": "n03916031",
    "superclass": "cosmetics",
    "class": "perfume"
  },
  {
    "id": "n03920288",
    "superclass": "lab equipment",
    "class": "Petri dish"
  },
  {
    "id": "n03924679",
    "superclass": "electronics",
    "class": "photocopier"
  },
  {
    "id": "n03929660",
    "superclass": "musical instrument",
    "class": "pick"
  },
  {
    "id": "n03929855",
    "superclass": "hat",
    "class": "pickelhaube"
  },
  {
    "id": "n03930313",
    "superclass": "fence",
    "class": "picket fence"
  },
  {
    "id": "n03930630",
    "superclass": "vehicle",
    "class": "pickup"
  },
  {
    "id": "n03933933",
    "superclass": "outdoor scene",
    "class": "pier"
  },
  {
    "id": "n03935335",
    "superclass": "decor",
    "class": "piggy bank"
  },
  {
    "id": "n03937543",
    "superclass": "container",
    "class": "pill bottle"
  },
  {
    "id": "n03938244",
    "superclass": "decor",
    "class": "pillow"
  },
  {
    "id": "n03942813",
    "superclass": "ball",
    "class": "ping-pong ball"
  },
  {
    "id": "n03944341",
    "superclass": "toy",
    "class": "pinwheel"
  },
  {
    "id": "n03947888",
    "superclass": "person",
    "class": "pirate"
  },
  {
    "id": "n03950228",
    "superclass": "kitchenware",
    "class": "pitcher"
  },
  {
    "id": "n03954731",
    "superclass": "tool",
    "class": "plane"
  },
  {
    "id": "n03956157",
    "superclass": "building",
    "class": "planetarium"
  },
  {
    "id": "n03958227",
    "superclass": "container",
    "class": "plastic bag"
  },
  {
    "id": "n03961711",
    "superclass": "decor",
    "class": "plate rack"
  },
  {
    "id": "n03967562",
    "superclass": "agricultural tool",
    "class": "plow"
  },
  {
    "id": "n03970156",
    "superclass": "tool",
    "class": "plunger"
  },
  {
    "id": "n03976467",
    "superclass": "photography equipment",
    "class": "Polaroid camera"
  },
  {
    "id": "n03976657",
    "superclass": "sports equipment",
    "class": "pole"
  },
  {
    "id": "n03977966",
    "superclass": "vehicle",
    "class": "police van"
  },
  {
    "id": "n03980874",
    "superclass": "clothing",
    "class": "poncho"
  },
  {
    "id": "n03982430",
    "superclass": "furniture",
    "class": "pool table"
  },
  {
    "id": "n03983396",
    "superclass": "food",
    "class": "pop bottle"
  },
  {
    "id": "n03991062",
    "superclass": "kitchenware",
    "class": "pot"
  },
  {
    "id": "n03992509",
    "superclass": "tool",
    "class": "potter's wheel"
  },
  {
    "id": "n03995372",
    "superclass": "electronics",
    "class": "power drill"
  },
  {
    "id": "n03998194",
    "superclass": "decor",
    "class": "prayer rug"
  },
  {
    "id": "n04004767",
    "superclass": "electronics",
    "class": "printer"
  },
  {
    "id": "n04005630",
    "superclass": "place",
    "class": "prison"
  },
  {
    "id": "n04008634",
    "superclass": "weapon",
    "class": "projectile"
  },
  {
    "id": "n04009552",
    "superclass": "electronics",
    "class": "projector"
  },
  {
    "id": "n04019541",
    "superclass": "sports equipment",
    "class": "puck"
  },
  {
    "id": "n04023962",
    "superclass": "sports equipment",
    "class": "punching bag"
  },
  {
    "id": "n04026417",
    "superclass": "accessory",
    "class": "purse"
  },
  {
    "id": "n04033901",
    "superclass": "tool",
    "class": "quill"
  },
  {
    "id": "n04033995",
    "superclass": "decor",
    "class": "quilt"
  },
  {
    "id": "n04037443",
    "superclass": "vehicle",
    "class": "racer"
  },
  {
    "id": "n04039381",
    "superclass": "sports equipment",
    "class": "racket"
  },
  {
    "id": "n04040759",
    "superclass": "electronics",
    "class": "radiator"
  },
  {
    "id": "n04041544",
    "superclass": "electronics",
    "class": "radio"
  },
  {
    "id": "n04044716",
    "superclass": "technology",
    "class": "radio telescope"
  },
  {
    "id": "n04049303",
    "superclass": "container",
    "class": "rain barrel"
  },
  {
    "id": "n04065272",
    "superclass": "vehicle",
    "class": "recreational vehicle"
  },
  {
    "id": "n04067472",
    "superclass": "technology",
    "class": "reel"
  },
  {
    "id": "n04069434",
    "superclass": "photography equipment",
    "class": "reflex camera"
  },
  {
    "id": "n04070727",
    "superclass": "furniture",
    "class": "refrigerator"
  },
  {
    "id": "n04074963",
    "superclass": "electronics",
    "class": "remote control"
  },
  {
    "id": "n04081281",
    "superclass": "place",
    "class": "restaurant"
  },
  {
    "id": "n04086273",
    "superclass": "weapon",
    "class": "revolver"
  },
  {
    "id": "n04090263",
    "superclass": "weapon",
    "class": "rifle"
  },
  {
    "id": "n04099969",
    "superclass": "furniture",
    "class": "rocking chair"
  },
  {
    "id": "n04111531",
    "superclass": "cooking",
    "class": "rotisserie"
  },
  {
    "id": "n04116512",
    "superclass": "tool",
    "class": "rubber eraser"
  },
  {
    "id": "n04118538",
    "superclass": "ball",
    "class": "rugby ball"
  },
  {
    "id": "n04118776",
    "superclass": "tool",
    "class": "rule"
  },
  {
    "id": "n04120489",
    "superclass": "clothing",
    "class": "running shoe"
  },
  {
    "id": "n04125021",
    "superclass": "container",
    "class": "safe"
  },
  {
    "id": "n04127249",
    "superclass": "tool",
    "class": "safety pin"
  },
  {
    "id": "n04131690",
    "superclass": "food",
    "class": "saltshaker"
  },
  {
    "id": "n04133789",
    "superclass": "clothing",
    "class": "sandal"
  },
  {
    "id": "n04136333",
    "superclass": "clothing",
    "class": "sarong"
  },
  {
    "id": "n04141076",
    "superclass": "instrument",
    "class": "sax"
  },
  {
    "id": "n04141327",
    "superclass": "weapon",
    "class": "scabbard"
  },
  {
    "id": "n04141975",
    "superclass": "electronics",
    "class": "scale"
  },
  {
    "id": "n04146614",
    "superclass": "vehicle",
    "class": "school bus"
  },
  {
    "id": "n04147183",
    "superclass": "boat",
    "class": "schooner"
  },
  {
    "id": "n04149813",
    "superclass": "sports equipment",
    "class": "scoreboard"
  },
  {
    "id": "n04152593",
    "superclass": "decor",
    "class": "screen"
  },
  {
    "id": "n04153751",
    "superclass": "tool",
    "class": "screw"
  },
  {
    "id": "n04154565",
    "superclass": "tool",
    "class": "screwdriver"
  },
  {
    "id": "n04162706",
    "superclass": "technology",
    "class": "seat belt"
  },
  {
    "id": "n04179913",
    "superclass": "technology",
    "class": "sewing machine"
  },
  {
    "id": "n04192698",
    "superclass": "tool",
    "class": "shield"
  },
  {
    "id": "n04200800",
    "superclass": "place",
    "class": "shoe shop"
  },
  {
    "id": "n04201297",
    "superclass": "furniture",
    "class": "shoji"
  },
  {
    "id": "n04204238",
    "superclass": "container",
    "class": "shopping basket"
  },
  {
    "id": "n04204347",
    "superclass": "container",
    "class": "shopping cart"
  },
  {
    "id": "n04208210",
    "superclass": "tool",
    "class": "shovel"
  },
  {
    "id": "n04209133",
    "superclass": "hat",
    "class": "shower cap"
  },
  {
    "id": "n04209239",
    "superclass": "decor",
    "class": "shower curtain"
  },
  {
    "id": "n04228054",
    "superclass": "sports equipment",
    "class": "ski"
  },
  {
    "id": "n04229816",
    "superclass": "clothing",
    "class": "ski mask"
  },
  {
    "id": "n04235860",
    "superclass": "accessory",
    "class": "sleeping bag"
  },
  {
    "id": "n04238763",
    "superclass": "technology",
    "class": "slide rule"
  },
  {
    "id": "n04239074",
    "superclass": "furniture",
    "class": "sliding door"
  },
  {
    "id": "n04243546",
    "superclass": "gaming",
    "class": "slot"
  },
  {
    "id": "n04251144",
    "superclass": "sports equipment",
    "class": "snorkel"
  },
  {
    "id": "n04252077",
    "superclass": "vehicle",
    "class": "snowmobile"
  },
  {
    "id": "n04252225",
    "superclass": "vehicle",
    "class": "snowplow"
  },
  {
    "id": "n04254120",
    "superclass": "decor",
    "class": "soap dispenser"
  },
  {
    "id": "n04254680",
    "superclass": "ball",
    "class": "soccer ball"
  },
  {
    "id": "n04254777",
    "superclass": "clothing",
    "class": "sock"
  },
  {
    "id": "n04258138",
    "superclass": "technology",
    "class": "solar dish"
  },
  {
    "id": "n04259630",
    "superclass": "hat",
    "class": "sombrero"
  },
  {
    "id": "n04263257",
    "superclass": "cooking",
    "class": "soup bowl"
  },
  {
    "id": "n04264628",
    "superclass": "electronics",
    "class": "space bar"
  },
  {
    "id": "n04265275",
    "superclass": "electronics",
    "class": "space heater"
  },
  {
    "id": "n04266014",
    "superclass": "aircraft",
    "class": "space shuttle"
  },
  {
    "id": "n04270147",
    "superclass": "cooking",
    "class": "spatula"
  },
  {
    "id": "n04273569",
    "superclass": "boat",
    "class": "speedboat"
  },
  {
    "id": "n04275548",
    "superclass": "outdoor scene",
    "class": "spider web"
  },
  {
    "id": "n04277352",
    "superclass": "tool",
    "class": "spindle"
  },
  {
    "id": "n04285008",
    "superclass": "vehicle",
    "class": "sports car"
  },
  {
    "id": "n04286575",
    "superclass": "electronics",
    "class": "spotlight"
  },
  {
    "id": "n04296562",
    "superclass": "furniture",
    "class": "stage"
  },
  {
    "id": "n04310018",
    "superclass": "train",
    "class": "steam locomotive"
  },
  {
    "id": "n04311004",
    "superclass": "outdoor scene",
    "class": "steel arch bridge"
  },
  {
    "id": "n04311174",
    "superclass": "instrument",
    "class": "steel drum"
  },
  {
    "id": "n04317175",
    "superclass": "tool",
    "class": "stethoscope"
  },
  {
    "id": "n04325704",
    "superclass": "accessory",
    "class": "stole"
  },
  {
    "id": "n04326547",
    "superclass": "outdoor scene",
    "class": "stone wall"
  },
  {
    "id": "n04328186",
    "superclass": "electronics",
    "class": "stopwatch"
  },
  {
    "id": "n04330267",
    "superclass": "furniture",
    "class": "stove"
  },
  {
    "id": "n04332243",
    "superclass": "cooking",
    "class": "strainer"
  },
  {
    "id": "n04335435",
    "superclass": "vehicle",
    "class": "streetcar"
  },
  {
    "id": "n04336792",
    "superclass": "furniture",
    "class": "stretcher"
  },
  {
    "id": "n04344873",
    "superclass": "furniture",
    "class": "studio couch"
  },
  {
    "id": "n04346328",
    "superclass": "building",
    "class": "stupa"
  },
  {
    "id": "n04347754",
    "superclass": "boat",
    "class": "submarine"
  },
  {
    "id": "n04350905",
    "superclass": "clothing",
    "class": "suit"
  },
  {
    "id": "n04355338",
    "superclass": "outdoor scene",
    "class": "sundial"
  },
  {
    "id": "n04355933",
    "superclass": "tool",
    "class": "sunglass"
  },
  {
    "id": "n04356056",
    "superclass": "clothing",
    "class": "sunglasses"
  },
  {
    "id": "n04357314",
    "superclass": "skin care product",
    "class": "sunscreen"
  },
  {
    "id": "n04366367",
    "superclass": "outdoor scene",
    "class": "suspension bridge"
  },
  {
    "id": "n04367480",
    "superclass": "tool",
    "class": "swab"
  },
  {
    "id": "n04370456",
    "superclass": "clothing",
    "class": "sweatshirt"
  },
  {
    "id": "n04371430",
    "superclass": "clothing",
    "class": "swimming trunks"
  },
  {
    "id": "n04371774",
    "superclass": "toy",
    "class": "swing"
  },
  {
    "id": "n04372370",
    "superclass": "electronics",
    "class": "switch"
  },
  {
    "id": "n04376876",
    "superclass": "tool",
    "class": "syringe"
  },
  {
    "id": "n04380533",
    "superclass": "decor",
    "class": "table lamp"
  },
  {
    "id": "n04389033",
    "superclass": "vehicle",
    "class": "tank"
  },
  {
    "id": "n04392985",
    "superclass": "electronics",
    "class": "tape player"
  },
  {
    "id": "n04398044",
    "superclass": "cooking",
    "class": "teapot"
  },
  {
    "id": "n04399382",
    "superclass": "toy",
    "class": "teddy"
  },
  {
    "id": "n04404412",
    "superclass": "electronics",
    "class": "television"
  },
  {
    "id": "n04409515",
    "superclass": "ball",
    "class": "tennis ball"
  },
  {
    "id": "n04417672",
    "superclass": "building",
    "class": "thatch"
  },
  {
    "id": "n04418357",
    "superclass": "decor",
    "class": "theater curtain"
  },
  {
    "id": "n04423845",
    "superclass": "tool",
    "class": "thimble"
  },
  {
    "id": "n04428191",
    "superclass": "vehicle",
    "class": "thresher"
  },
  {
    "id": "n04429376",
    "superclass": "furniture",
    "class": "throne"
  },
  {
    "id": "n04435653",
    "superclass": "building",
    "class": "tile roof"
  },
  {
    "id": "n04442312",
    "superclass": "electronics",
    "class": "toaster"
  },
  {
    "id": "n04443257",
    "superclass": "okace",
    "class": "tobacco shop"
  },
  {
    "id": "n04447861",
    "superclass": "furniture",
    "class": "toilet seat"
  },
  {
    "id": "n04456115",
    "superclass": "tool",
    "class": "torch"
  },
  {
    "id": "n04458633",
    "superclass": "outdoor scene",
    "class": "totem pole"
  },
  {
    "id": "n04461696",
    "superclass": "vehicle",
    "class": "tow truck"
  },
  {
    "id": "n04462240",
    "superclass": "place",
    "class": "toyshop"
  },
  {
    "id": "n04465501",
    "superclass": "vehicle",
    "class": "tractor"
  },
  {
    "id": "n04467665",
    "superclass": "vehicle",
    "class": "trailer truck"
  },
  {
    "id": "n04476259",
    "superclass": "decor",
    "class": "tray"
  },
  {
    "id": "n04479046",
    "superclass": "clothing",
    "class": "trench coat"
  },
  {
    "id": "n04482393",
    "superclass": "vehicle",
    "class": "tricycle"
  },
  {
    "id": "n04483307",
    "superclass": "boat",
    "class": "trimaran"
  },
  {
    "id": "n04485082",
    "superclass": "technology",
    "class": "tripod"
  },
  {
    "id": "n04486054",
    "superclass": "outdoor scene",
    "class": "triumphal arch"
  },
  {
    "id": "n04487081",
    "superclass": "vehicle",
    "class": "trolleybus"
  },
  {
    "id": "n04487394",
    "superclass": "instrument",
    "class": "trombone"
  },
  {
    "id": "n04493381",
    "superclass": "bathroom fixture",
    "class": "tub"
  },
  {
    "id": "n04501370",
    "superclass": "technology",
    "class": "turnstile"
  },
  {
    "id": "n04505470",
    "superclass": "electronics",
    "class": "typewriter keyboard"
  },
  {
    "id": "n04507155",
    "superclass": "accessory",
    "class": "umbrella"
  },
  {
    "id": "n04509417",
    "superclass": "vehicle",
    "class": "unicycle"
  },
  {
    "id": "n04515003",
    "superclass": "instrument",
    "class": "upright"
  },
  {
    "id": "n04517823",
    "superclass": "home appliance",
    "class": "vacuum"
  },
  {
    "id": "n04522168",
    "superclass": "decor",
    "class": "vase"
  },
  {
    "id": "n04523525",
    "superclass": "architectural component",
    "class": "vault"
  },
  {
    "id": "n04525038",
    "superclass": "fabric",
    "class": "velvet"
  },
  {
    "id": "n04525305",
    "superclass": "technology",
    "class": "vending machine"
  },
  {
    "id": "n04532106",
    "superclass": "clothing",
    "class": "vestment"
  },
  {
    "id": "n04532670",
    "superclass": "outdoor scene",
    "class": "viaduct"
  },
  {
    "id": "n04536866",
    "superclass": "instrument",
    "class": "violin"
  },
  {
    "id": "n04540053",
    "superclass": "ball",
    "class": "volleyball"
  },
  {
    "id": "n04542943",
    "superclass": "electronics",
    "class": "waffle iron"
  },
  {
    "id": "n04548280",
    "superclass": "decor",
    "class": "wall clock"
  },
  {
    "id": "n04548362",
    "superclass": "accessory",
    "class": "wallet"
  },
  {
    "id": "n04550184",
    "superclass": "furniture",
    "class": "wardrobe"
  },
  {
    "id": "n04552348",
    "superclass": "aircraft",
    "class": "warplane"
  },
  {
    "id": "n04553703",
    "superclass": "container",
    "class": "washbasin"
  },
  {
    "id": "n04554684",
    "superclass": "home appliance",
    "class": "washer"
  },
  {
    "id": "n04557648",
    "superclass": "container",
    "class": "water bottle"
  },
  {
    "id": "n04560804",
    "superclass": "container",
    "class": "water jug"
  },
  {
    "id": "n04562935",
    "superclass": "outdoor scene",
    "class": "water tower"
  },
  {
    "id": "n04579145",
    "superclass": "container",
    "class": "whiskey jug"
  },
  {
    "id": "n04579432",
    "superclass": "tool",
    "class": "whistle"
  },
  {
    "id": "n04584207",
    "superclass": "clothing",
    "class": "wig"
  },
  {
    "id": "n04589890",
    "superclass": "decor",
    "class": "window screen"
  },
  {
    "id": "n04590129",
    "superclass": "decor",
    "class": "window shade"
  },
  {
    "id": "n04591157",
    "superclass": "accessory",
    "class": "Windsor tie"
  },
  {
    "id": "n04591713",
    "superclass": "food",
    "class": "wine bottle"
  },
  {
    "id": "n04592741",
    "superclass": "aircraft component",
    "class": "wing"
  },
  {
    "id": "n04596742",
    "superclass": "kitchenware",
    "class": "wok"
  },
  {
    "id": "n04597913",
    "superclass": "cooking",
    "class": "wooden spoon"
  },
  {
    "id": "n04599235",
    "superclass": "other",
    "class": "wool"
  },
  {
    "id": "n04604644",
    "superclass": "fence",
    "class": "worm fence"
  },
  {
    "id": "n04606251",
    "superclass": "boat",
    "class": "wreck"
  },
  {
    "id": "n04612504",
    "superclass": "boat",
    "class": "yawl"
  },
  {
    "id": "n04613696",
    "superclass": "building",
    "class": "yurt"
  },
  {
    "id": "n06359193",
    "superclass": "technology",
    "class": "web site"
  },
  {
    "id": "n06596364",
    "superclass": "paper",
    "class": "comic book"
  },
  {
    "id": "n06785654",
    "superclass": "paper",
    "class": "crossword puzzle"
  },
  {
    "id": "n06794110",
    "superclass": "outdoor scene",
    "class": "street sign"
  },
  {
    "id": "n06874185",
    "superclass": "electronics",
    "class": "traffic light"
  },
  {
    "id": "n07248320",
    "superclass": "paper",
    "class": "book jacket"
  },
  {
    "id": "n07565083",
    "superclass": "paper",
    "class": "menu"
  },
  {
    "id": "n07579787",
    "superclass": "cooking",
    "class": "plate"
  },
  {
    "id": "n07583066",
    "superclass": "food",
    "class": "guacamole"
  },
  {
    "id": "n07584110",
    "superclass": "food",
    "class": "consomme"
  },
  {
    "id": "n07590611",
    "superclass": "cooking",
    "class": "hot pot"
  },
  {
    "id": "n07613480",
    "superclass": "food",
    "class": "trifle"
  },
  {
    "id": "n07614500",
    "superclass": "food",
    "class": "ice cream"
  },
  {
    "id": "n07615774",
    "superclass": "food",
    "class": "ice lolly"
  },
  {
    "id": "n07684084",
    "superclass": "food",
    "class": "French loaf"
  },
  {
    "id": "n07693725",
    "superclass": "food",
    "class": "bagel"
  },
  {
    "id": "n07695742",
    "superclass": "food",
    "class": "pretzel"
  },
  {
    "id": "n07697313",
    "superclass": "food",
    "class": "cheeseburger"
  },
  {
    "id": "n07697537",
    "superclass": "food",
    "class": "hotdog"
  },
  {
    "id": "n07711569",
    "superclass": "food",
    "class": "mashed potato"
  },
  {
    "id": "n07714571",
    "superclass": "vegetable",
    "class": "head cabbage"
  },
  {
    "id": "n07714990",
    "superclass": "vegetable",
    "class": "broccoli"
  },
  {
    "id": "n07715103",
    "superclass": "vegetable",
    "class": "cauliflower"
  },
  {
    "id": "n07716358",
    "superclass": "fruit",
    "class": "zucchini"
  },
  {
    "id": "n07716906",
    "superclass": "fruit",
    "class": "spaghetti squash"
  },
  {
    "id": "n07717410",
    "superclass": "fruit",
    "class": "acorn squash"
  },
  {
    "id": "n07717556",
    "superclass": "vegetable",
    "class": "butternut squash"
  },
  {
    "id": "n07718472",
    "superclass": "vegetable",
    "class": "cucumber"
  },
  {
    "id": "n07718747",
    "superclass": "vegetable",
    "class": "artichoke"
  },
  {
    "id": "n07720875",
    "superclass": "vegetable",
    "class": "bell pepper"
  },
  {
    "id": "n07730033",
    "superclass": "flower",
    "class": "cardoon"
  },
  {
    "id": "n07734744",
    "superclass": "fungus",
    "class": "mushroom"
  },
  {
    "id": "n07742313",
    "superclass": "fruit",
    "class": "Granny Smith"
  },
  {
    "id": "n07745940",
    "superclass": "fruit",
    "class": "strawberry"
  },
  {
    "id": "n07747607",
    "superclass": "fruit",
    "class": "orange"
  },
  {
    "id": "n07749582",
    "superclass": "fruit",
    "class": "lemon"
  },
  {
    "id": "n07753113",
    "superclass": "fruit",
    "class": "fig"
  },
  {
    "id": "n07753275",
    "superclass": "fruit",
    "class": "pineapple"
  },
  {
    "id": "n07753592",
    "superclass": "fruit",
    "class": "banana"
  },
  {
    "id": "n07754684",
    "superclass": "fruit",
    "class": "jackfruit"
  },
  {
    "id": "n07760859",
    "superclass": "fruit",
    "class": "custard apple"
  },
  {
    "id": "n07768694",
    "superclass": "fruit",
    "class": "pomegranate"
  },
  {
    "id": "n07802026",
    "superclass": "agricultural product",
    "class": "hay"
  },
  {
    "id": "n07831146",
    "superclass": "food",
    "class": "carbonara"
  },
  {
    "id": "n07836838",
    "superclass": "food",
    "class": "chocolate sauce"
  },
  {
    "id": "n07860988",
    "superclass": "food",
    "class": "dough"
  },
  {
    "id": "n07871810",
    "superclass": "food",
    "class": "meat loaf"
  },
  {
    "id": "n07873807",
    "superclass": "food",
    "class": "pizza"
  },
  {
    "id": "n07875152",
    "superclass": "food",
    "class": "potpie"
  },
  {
    "id": "n07880968",
    "superclass": "food",
    "class": "burrito"
  },
  {
    "id": "n07892512",
    "superclass": "food",
    "class": "red wine"
  },
  {
    "id": "n07920052",
    "superclass": "food",
    "class": "espresso"
  },
  {
    "id": "n07930864",
    "superclass": "kitchenware",
    "class": "cup"
  },
  {
    "id": "n07932039",
    "superclass": "food",
    "class": "eggnog"
  },
  {
    "id": "n09193705",
    "superclass": "outdoor scene",
    "class": "alp"
  },
  {
    "id": "n09229709",
    "superclass": "toy",
    "class": "bubble"
  },
  {
    "id": "n09246464",
    "superclass": "outdoor scene",
    "class": "cliff"
  },
  {
    "id": "n09256479",
    "superclass": "coral",
    "class": "coral reef"
  },
  {
    "id": "n09288635",
    "superclass": "outdoor scene",
    "class": "geyser"
  },
  {
    "id": "n09332890",
    "superclass": "outdoor scene",
    "class": "lakeside"
  },
  {
    "id": "n09399592",
    "superclass": "outdoor scene",
    "class": "promontory"
  },
  {
    "id": "n09421951",
    "superclass": "outdoor scene",
    "class": "sandbar"
  },
  {
    "id": "n09428293",
    "superclass": "outdoor scene",
    "class": "seashore"
  },
  {
    "id": "n09468604",
    "superclass": "outdoor scene",
    "class": "valley"
  },
  {
    "id": "n09472597",
    "superclass": "outdoor scene",
    "class": "volcano"
  },
  {
    "id": "n09835506",
    "superclass": "person",
    "class": "ballplayer"
  },
  {
    "id": "n10148035",
    "superclass": "person",
    "class": "groom"
  },
  {
    "id": "n10565667",
    "superclass": "person",
    "class": "scuba diver"
  },
  {
    "id": "n11879895",
    "superclass": "flower",
    "class": "rapeseed"
  },
  {
    "id": "n11939491",
    "superclass": "flower",
    "class": "daisy"
  },
  {
    "id": "n12057211",
    "superclass": "flower",
    "class": "yellow lady's slipper"
  },
  {
    "id": "n12144580",
    "superclass": "food",
    "class": "corn"
  },
  {
    "id": "n12267677",
    "superclass": "plant",
    "class": "acorn"
  },
  {
    "id": "n12620546",
    "superclass": "flower",
    "class": "hip"
  },
  {
    "id": "n12768682",
    "superclass": "plant",
    "class": "buckeye"
  },
  {
    "id": "n12985857",
    "superclass": "coral",
    "class": "coral fungus"
  },
  {
    "id": "n12998815",
    "superclass": "fungus",
    "class": "agaric"
  },
  {
    "id": "n13037406",
    "superclass": "fungus",
    "class": "gyromitra"
  },
  {
    "id": "n13040303",
    "superclass": "fungus",
    "class": "stinkhorn"
  },
  {
    "id": "n13044778",
    "superclass": "fungus",
    "class": "earthstar"
  },
  {
    "id": "n13052670",
    "superclass": "fungus",
    "class": "hen-of-the-woods"
  },
  {
    "id": "n13054560",
    "superclass": "fungus",
    "class": "bolete"
  },
  {
    "id": "n13133613",
    "superclass": "food",
    "class": "ear"
  },
  {
    "id": "n15075141",
    "superclass": "toilet supply",
    "class": "toilet tissue"
  }
]

def make_imagefolder(path):
    filelist = os.listdir(path)
    filelist_png = [file for file in filelist if file.endswith(".png")]
    for filename in filelist_png:
        filename_only, png = filename.split(".")
        it, idx, prompt = filename_only.split("_")
        class_name = prompt.replace("a photo of ", "")
        folder = ''
        for i in range(len(class_list)):
            if (class_name == class_list[i][2]):
                folder = class_list[i][0]
            else:
                raise Exception("Wrong class")

        newpath = path + folder + "/"
        if not os.path.isdir(newpath):
            os.makedirs(newpath)
        shutil.copy(os.path.join(path, filename), os.path.join(newpath, filename))

def make_imagefolder_superclass(path):
    filelist = os.listdir(path)
    filelist_png = [file for file in filelist if file.endswith(".png")]
    for filename in filelist_png:
        filename_only, png = filename.split(".")
        it, idx, prompt = filename_only.split("_")
        class_name, superclass_name = prompt.split(" a type of ")
        class_name = class_name.replace("a photo of ", "")
        class_name = class_name.lower()

        folder = ''
        for i in range(len(class_list)):
            if (class_name == class_list[i][2]):
                folder = class_list[i][0]

        if folder == '':
            raise Exception("Wrong class name: " + class_name)

        newpath = path + folder + "/"

        if not os.path.isdir(newpath):
            os.makedirs(newpath)
        shutil.copy(os.path.join(path, filename), os.path.join(newpath, filename))



def make_imagefolder_superclass_ldmv2(path):
    filelist = os.listdir(path)
    filelist_png = [file for file in filelist if file.endswith(".png")]
    for filename in filelist_png:

        if '_2.png' in filename:
            filename2 = filename.replace('_2.png','.png')
        elif '_3.png' in filename:
            filename2 = filename.replace('_3.png', '.png')
        elif '_4.png' in filename:
            filename2 = filename.replace('_4.png','.png')
        else:
            filename2= filename

        filename_only, *_ = filename2.split(".")
        it, idx, prompt = filename_only.split("_")
        if 'the type of' in prompt:
            class_name, superclass_name = prompt.split(" a ty")
        else:
            class_name = prompt
        class_name = class_name.replace("a photo of ", "")
        if ',' in class_name:
            class_name, *_ = class_name.split(',')
        else:
            class_name = class_name
        class_name = class_name.lower()

        if ' the ' in class_name:
            class_name, *_ = class_name.split(' the ')

        folder = ''
        for i in range(len(class_list)):
            if (class_name == class_list[i][2]):
                folder = class_list[i][0]

        if folder == '':
            raise Exception("Wrong class name: " + class_name)

        newpath = path + folder + "/"

        if not os.path.isdir(newpath):
            os.makedirs(newpath)
        shutil.copy(os.path.join(path, filename), os.path.join(newpath, filename))

def make_imagefolder_description(path):
    filelist = os.listdir(path)
    filelist_png = [file for file in filelist if file.endswith(".png")]
    for filename in filelist_png:
        filename_only= filename.replace(".png",'')
        it, idx, prompt = filename_only.split("_",2)
        try:
          class_name, superclass_name = prompt.split(" a type of ",1)
        except:
          print(filename_only, prompt)
        class_name = class_name.replace("A photo of ", "")
        class_name = class_name.replace("a photo of ","")
        class_name = class_name.replace(',',"")
        class_name = class_name.lower()
        class_name = class_name.replace('a ','',1)
        class_name = class_name.strip()

        folder = ''
        for i in range(len(class_list)):
            if (class_list[i]['class'].lower() == class_name and class_list[i]['superclass'].lower() == superclass_name):
                folder = class_list[i]['id']

        if folder == '':
            raise Exception("Wrong class name: " + class_name)

        newpath = path + folder + "/"

        if not os.path.isdir(newpath):
            os.makedirs(newpath)
        shutil.move(os.path.join(path, filename), os.path.join(newpath, filename))