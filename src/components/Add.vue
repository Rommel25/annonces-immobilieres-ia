<template>
  <div>
    <v-card class="mx-auto" max-width="600">
      <v-card-title>
        <h3>Ajouter un nouvel appartement</h3>
      </v-card-title>
      <v-card-text>
        <form @submit.prevent="ajouterAppartement">
          <div>
            <v-text-field label="Nombre de chambres" v-model="nouvelAppartement.nbRooms" type="number" id="nbRooms" required />
          </div>
          <div>
            <v-text-field label="Surface (m²)" v-model="nouvelAppartement.surface" type="number" id="surface" required />
          </div>
          <div>
            <v-text-field label="Nombre de fenêtres" v-model="nouvelAppartement.nbWindows" type="number" id="nbWindows" required />
          </div>
          <div>
            <v-text-field label="Prix" v-model="nouvelAppartement.price" type="number" id="price" required />
          </div>
          <div>
            <v-select
              label="Année de construction"
              :items="anneesOptions"
              v-model="nouvelAppartement.annee"
              required
            ></v-select>
          </div>
          <div>
            <v-checkbox label="Balcon" v-model="nouvelAppartement.balcon"></v-checkbox>
          </div>
          <div>
            <v-checkbox label="Garage" v-model="nouvelAppartement.garage"></v-checkbox>
          </div>
          <div>
            <v-select
              label="Note"
              :items="[1, 2, 3, 4, 5]"
              v-model="nouvelAppartement.note"
              required
            ></v-select>
          </div>
          <div>
            <v-select
              label="Catégorie de prix"
              :items="['low', 'normal', 'high', 'scam']"
              v-model="nouvelAppartement.price_category"
              required
            ></v-select>
          </div>
          <v-btn type="submit">Ajouter l'appartement</v-btn>
        </form>
      </v-card-text>
    </v-card>
  </div>
</template>

<script>
import { defineComponent, reactive } from 'vue';

export default defineComponent({
  name: "AjoutAppartement",
  emits: ['appartement-ajoute'],
  setup(props, { emit }) {
    // Options pour l'année de construction entre 2005 et 2024
    const anneesOptions = Array.from({ length: 2024 - 2005 + 1 }, (v, i) => 2005 + i);

    const nouvelAppartement = reactive({
      nbRooms: 4,
      surface: 20,
      nbWindows: 3,
      price: 100000,
      annee: 2024,  // Par défaut l'année la plus récente
      balcon: false,  // Par défaut, pas de balcon
      garage: false,  // Par défaut, pas de garage
      note: 3,  // Par défaut, note moyenne
      price_category: 'normal'  // Par défaut, prix normal
    });

    const ajouterAppartement = () => {
      // Générer un ID unique pour le nouvel appartement
      const nouvelId = Date.now();
      const appartementComplet = { ...nouvelAppartement, id: nouvelId };
      emit('appartement-ajoute', appartementComplet);

      // Réinitialiser le formulaire
      nouvelAppartement.nbRooms = 0;
      nouvelAppartement.surface = 0;
      nouvelAppartement.nbWindows = 0;
      nouvelAppartement.price = 0;
      nouvelAppartement.annee = 2024;
      nouvelAppartement.balcon = false;
      nouvelAppartement.garage = false;
      nouvelAppartement.note = 3;
      nouvelAppartement.price_category = 'normal';
    };

    return { nouvelAppartement, ajouterAppartement, anneesOptions };
  }
});
</script>
